# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor runtime.

``SpecializedDecode`` runs the decoder layers of ``DeepseekV4Model`` for the
steps captured into FULL CUDA graphs of pure decode batches of up to
``AMPERE_MAX_ROWS`` rows, preserving 100% mathematical equivalence.
"""

from typing import TYPE_CHECKING, NamedTuple, TypeAlias, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.models.deepseek_v4_1.common.engram import Engram
from vllm.sequence import IntermediateTensors

from .. import AMPERE_MAX_ROWS, DecodeConfig

if TYPE_CHECKING:
    from vllm.models.deepseek_v4_1.nvidia.model import (
        DeepseekV4DecoderLayer,
        DeepseekV4Model,
    )
    from ..ampere.steps import AmpereAttentionStep, AmpereMoEStep
    from ..boundary.boundary import AmpereBoundaries

DecodeOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]


class DecodeStep(NamedTuple):
    """The host-side gates of one captured decode step, by its rows."""
    rows: int
    side_stream: bool
    low_batch: bool
    overlap_inputs: bool
    engram_prefetch: bool
    executor: "SpecializedDecode"

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
    ) -> DecodeOutput:
        return self.executor.run(
            self, hidden_states, positions, input_ids, engram_hashes, engram_mask
        )


class LayerPlan(NamedTuple):
    layer_id: int
    layer: "DeepseekV4DecoderLayer"
    attention: "AmpereAttentionStep"
    moe: "AmpereMoEStep"
    engram: Engram | None


class SpecializedDecode:
    """Base decode executor over one backbone's local layers."""

    def __init__(
        self,
        backbone: "DeepseekV4Model",
        config: DecodeConfig,
        lm_head: nn.Module,
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.lm_head = lm_head
        self.max_rows = AMPERE_MAX_ROWS
        self.plans: list[LayerPlan] = []
        self.boundaries: "AmpereBoundaries"

    @classmethod
    def for_target(
        cls, backbone: "DeepseekV4Model", config: DecodeConfig, lm_head: nn.Module
    ) -> "SpecializedDecode":
        from ..ampere.decode import AmpereSpecializedDecode
        return AmpereSpecializedDecode(backbone, config, lm_head)

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> DecodeStep | None:
        rows = positions.shape[0]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= self.max_rows
            or not is_forward_context_available()
        ):
            return None
        if not torch.cuda.is_current_stream_capturing():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
            
        return DecodeStep(
            rows=rows,
            side_stream=True,
            low_batch=rows <= 8,
            overlap_inputs=True,
            engram_prefetch=rows <= 16,
            executor=self,
        )

    def run(
        self,
        step: DecodeStep,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
    ) -> DecodeOutput:
        backbone = self.backbone
        aux_layers = backbone.aux_hidden_state_layers
        aux_hidden_states: list[torch.Tensor] = []
        residual, post_mix, res_mix, pre_mix = None, None, None, None
        
        hidden_states = x
        for idx, plan in enumerate(self.plans, start=backbone.start_layer):
            layer = plan.layer
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
            )
            if idx + 1 in aux_layers:
                aux_recon = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )
                aux_hidden_states.append(aux_recon.mean(dim=1))

        if not get_pp_group().is_last_rank:
            hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
            tensors = {"hidden_states": hidden_states, "pre_mix": pre_mix}
            tensors.update(backbone.pack_local_aux_hidden_states(aux_hidden_states))
            if backbone.pp_share is not None:
                tensors.update(
                    backbone.pp_share.send(
                        positions, backbone.topk_indices_buffer, backbone.candidate_block_buffer
                    )
                )
            return IntermediateTensors(tensors)

        # Last PP rank
        if backbone._mtp_hidden_buffer is not None:
            backbone._mtp_hidden_buffer[: hidden_states.shape[0]].copy_(hidden_states.flatten(1))
            
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        hidden_states = backbone.norm(hc_collapse_triton(hidden_states, pre_mix))
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
