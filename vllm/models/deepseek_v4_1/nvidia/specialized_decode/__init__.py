# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 specialized decode on NVIDIA Ampere (SM80) & Hopper (SM90) GPUs:
a decode executor over the stock backbone, and one over the DSpark draft model,
enabled by ``additional_config={"decode_backend": "latency_specialized"}``
or environment variable ``VLLM_SM80_SPECIALIZED_DECODE=1``.
"""

import os
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from ..dspark import DSparkDeepseekV4ForCausalLM

logger = init_logger(__name__)

AMPERE_MAX_ROWS = 192
HOPPER_MAX_ROWS = 192


def max_rows() -> int:
    """Rows of the largest step this GPU's executor runs."""
    return AMPERE_MAX_ROWS


@dataclass(frozen=True)
class DecodeConfig:
    autotune: bool = True
    jit_warmup: bool = True
    enable_sm80_moe_fastpath: bool = True
    enable_sm80_sparse_mla: bool = True
    enable_sm80_fused_dspark: bool = True


def decode_backend(vllm_config: VllmConfig) -> str:
    """Determine whether latency_specialized is requested."""
    if os.environ.get("VLLM_SM80_SPECIALIZED_DECODE", "0") in ("1", "true", "True"):
        return "latency_specialized"
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend not in ("auto", "latency_specialized"):
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    return backend


def unsupported(vllm_config: VllmConfig) -> list[str]:
    """Check requirements for SM80 specialized decode."""
    capability = current_platform.get_device_capability()
    is_sm80 = capability is not None and capability.major == 8 and capability.minor == 0
    
    attention = vllm_config.attention_config
    cache_dtype = vllm_config.cache_config.cache_dtype
    attn_backend_ok = getattr(attention, "backend", None) in (None, "TRITON_MLA_SPARSE_DSV41")
    cache_dtype_ok = cache_dtype in ("auto", "fp8", "fp8_ds_mla")
    
    requirements = {
        "an Ampere SM80 (A100/A800) GPU": is_sm80,
        "the TRITON_MLA_SPARSE_DSV41 attention backend with fp8_ds_mla KV cache": (
            attn_backend_ok and cache_dtype_ok
        ),
        "tensor parallel size 8": vllm_config.parallel_config.tensor_parallel_size == 8,
        "FULL CUDA graph mode": (
            vllm_config.compilation_config.cudagraph_mode is not None
        ),
    }
    return [need for need, ok in requirements.items() if not ok]


def specialized_decode_config(vllm_config: VllmConfig) -> DecodeConfig | None:
    """Build DecodeConfig if requested and supported."""
    if decode_backend(vllm_config) == "auto":
        return None
    missing = unsupported(vllm_config)
    if missing:
        logger.warning_once(
            "DeepSeek-V4.1 specialized decode cannot run: missing %s",
            ", ".join(missing),
        )
        return None
    logger.info_once(
        "DeepSeek-V4.1 SM80 Native Specialized Decode enabled for FULL-graph steps up to %d rows.",
        max_rows(),
    )
    return DecodeConfig()


_drafts: list["weakref.ref[DSparkDeepseekV4ForCausalLM]"] = []


def register_draft(draft: "DSparkDeepseekV4ForCausalLM") -> None:
    _drafts.append(weakref.ref(draft))


def take_drafts() -> list["DSparkDeepseekV4ForCausalLM"]:
    drafts = [draft for ref in _drafts if (draft := ref()) is not None]
    _drafts.clear()
    return drafts
