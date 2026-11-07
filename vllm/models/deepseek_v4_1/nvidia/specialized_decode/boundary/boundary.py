# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mHC boundaries of the SM80 specialized decode.

Manages sublayer transitions (Attention <-> MoE), intra-node TP All-Reduce,
and mHC residual stream mixing with side-stream overlapping.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)

if TYPE_CHECKING:
    from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4DecoderLayer
    from ..runtime.decode import DecodeStep


@dataclass
class BoundaryState:
    """The hc residual streams and the mixes carried to the next boundary."""
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    pre: torch.Tensor
    pending: torch.Tensor | None = None


class AmpereBoundaries:
    """Sublayer boundaries for Ampere SM80 Specialized Decode."""

    def __init__(self, mhc_stream: torch.cuda.Stream | None = None) -> None:
        self.mhc_stream = mhc_stream

    def first(
        self,
        layer: "DeepseekV4DecoderLayer",
        x: torch.Tensor,
        step: "DecodeStep",
    ) -> tuple[BoundaryState, torch.Tensor]:
        """First layer entry: expand embedding across mHC streams."""
        residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
        fn = layer.hc_attn_fn_broadcast if getattr(layer, "hc_attn_fn_broadcast", None) is not None else layer.hc_attn_fn
        return self._pre(layer, residual, fn, None, x, step)
    def pre(
        self,
        layer: "DeepseekV4DecoderLayer",
        residual: torch.Tensor,
        state: BoundaryState,
        step: "DecodeStep",
    ) -> tuple[BoundaryState, torch.Tensor]:
        """Engram layer entry on injected streams."""
        return self._pre(layer, residual, layer.hc_attn_fn, state.pre, None, step)

    def sublayer(
        self,
        layer: "DeepseekV4DecoderLayer",
        ffn: bool,
        state: BoundaryState,
        step: "DecodeStep",
        capture_aux: bool = False,
    ) -> tuple[BoundaryState, torch.Tensor, torch.Tensor | None]:
        """Reduce pending tensor and prepare next sublayer input."""
        if ffn:
            fn, scale, base = layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base
            norm = layer.ffn_norm
        else:
            fn, scale, base = layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
            norm = layer.attn_norm

        assert state.pending is not None
        # 1. Reduce pending sublayer output across TP ranks
        x_reduced = tensor_model_parallel_all_reduce(state.pending)
        
        # 2. mHC Post-mixing into residual streams
        residual = mhc_post_tilelang(x_reduced, state.residual, state.post, state.comb)
        
        # 3. mHC Pre-mixing on side stream or current stream
        stream = self.mhc_stream if step.side_stream and self.mhc_stream is not None else torch.cuda.current_stream()
        
        post, comb, layer_input, next_pre = mhc_pre_delayed_tilelang(
            residual,
            fn,
            scale,
            base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=state.pre,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
        )
        
        aux = None
        if capture_aux:
            aux = residual.mean(dim=1)
            
        return BoundaryState(residual, post, comb, next_pre), layer_input, aux

    def post_only(self, state: BoundaryState) -> torch.Tensor:
        """Collapse streams for final layer norm or Engram injection."""
        assert state.pending is not None
        x_reduced = tensor_model_parallel_all_reduce(state.pending)
        return mhc_post_tilelang(x_reduced, state.residual, state.post, state.comb)

    def join(self, layer: "DeepseekV4DecoderLayer", step: "DecodeStep") -> None:
        """Synchronize side stream work."""
        if step.side_stream and self.mhc_stream is not None:
            torch.cuda.current_stream().wait_stream(self.mhc_stream)

    def _pre(
        self,
        layer: "DeepseekV4DecoderLayer",
        residual: torch.Tensor,
        fn: torch.Tensor,
        pre_mix: torch.Tensor | None,
        x: torch.Tensor | None,
        step: "DecodeStep",
    ) -> tuple[BoundaryState, torch.Tensor]:
        post, comb, layer_input, next_pre = mhc_pre_delayed_tilelang(
            residual,
            fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            x=x,
            norm_weight=layer.attn_norm.weight,
            norm_eps=layer.attn_norm.variance_epsilon,
        )
        return BoundaryState(residual, post, comb, next_pre), layer_input
