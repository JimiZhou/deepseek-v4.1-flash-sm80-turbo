import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context, ForwardContext
from vllm.models.deepseek_v4_1.nvidia.specialized_decode import DecodeConfig
from vllm.models.deepseek_v4_1.nvidia.specialized_decode.ampere.decode import AmpereSpecializedDecode
from vllm.models.deepseek_v4_1.nvidia.specialized_decode.runtime.decode import DecodeStep

class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = None
    def forward(self, positions, layer_input):
        return layer_input

class DummyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 6
        self.routed_scaling_factor = 1.5
        self.gate = nn.Linear(5120, 384, bias=False)
        self.experts = lambda x, w, idx: x
        self.shared_experts = None
    def forward(self, x):
        return x

class DummyLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.attn = DummyAttention()
        self.ffn = DummyMoE()
        self.engram = None
        self.mhc_mult = 4
        self.hc_mult = 4
        self.mhc_stream = None
        self.rms_norm_eps = 1e-6
        self.hc_eps = 1e-6
        self.hc_post_alpha = 1.0
        self.hc_sinkhorn_iters = 20
        self.attn_norm = nn.LayerNorm(5120)
        self.ffn_norm = nn.LayerNorm(5120)
        self.hc_attn_fn = torch.randn(4, 4, 5120)
        self.hc_ffn_fn = torch.randn(4, 4, 5120)
        self.hc_attn_scale = torch.ones(5120)
        self.hc_ffn_scale = torch.ones(5120)
        self.hc_attn_base = torch.zeros(5120)
        self.hc_ffn_base = torch.zeros(5120)

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_layer = 0
        self.end_layer = 2
        self.layers = nn.ModuleList([DummyLayer(0), DummyLayer(1)])
        self.aux_hidden_state_layers = []
        self._mtp_hidden_buffer = None
        self.norm = nn.LayerNorm(5120)

def test_executor_instantiation_and_gating():
    print("=" * 60)
    print("TESTING AMPERE SPECIALIZED DECODE EXECUTOR & GATING RUNTIME")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = DummyBackbone().to(device)
    config = DecodeConfig()
    lm_head = nn.Linear(5120, 1000).to(device)
    
    # 1. Instantiate AmpereSpecializedDecode
    executor = AmpereSpecializedDecode(backbone, config, lm_head)
    print("1. AmpereSpecializedDecode instantiated successfully.")
    assert len(executor.plans) == 2, "Expected 2 layer plans"
    
    # 2. Test Fallback on Prefill Shape (M=256 > max_rows 192)
    pos_prefill = torch.zeros(256, dtype=torch.int64, device=device)
    step_prefill = executor.plan(pos_prefill, None)
    assert step_prefill is None, f"Expected None for prefill, got {step_prefill}"
    print("2. Prefill fallback gating verified: returned None as expected.")
    
    # 3. Test Fallback in Eager Mode (not CUDA Graph capturing)
    pos_decode = torch.zeros(4, dtype=torch.int64, device=device)
    step_eager = executor.plan(pos_decode, None)
    assert step_eager is None, f"Expected None in eager mode, got {step_eager}"
    print("3. Eager fallback gating verified: returned None as expected.")
    
    # 4. Test DecodeStep Generation in CUDA Graph Capturing Mode
    # In CUDA Graph capture: torch.cuda.is_current_stream_capturing() is True
    # Let's test with a simulated graph capture stream
    if torch.cuda.is_available():
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        import unittest.mock
        step_captured = [None]
        with unittest.mock.patch("vllm.models.deepseek_v4_1.nvidia.specialized_decode.runtime.decode.is_forward_context_available", return_value=True), \
             unittest.mock.patch("vllm.models.deepseek_v4_1.nvidia.specialized_decode.runtime.decode.get_forward_context") as mock_ctx:
            mock_ctx.return_value.cudagraph_runtime_mode = CUDAGraphMode.NONE
            with torch.cuda.graph(g, stream=s):
                dummy_t = torch.zeros(1, device=device)
                step_captured[0] = executor.plan(pos_decode, None)
        print(f"4. Under CUDA Graph Capture: plan() returned {step_captured[0]}")
        assert isinstance(step_captured[0], DecodeStep), f"Expected DecodeStep, got {step_captured[0]}"
        assert step_captured[0].rows == 4
        assert step_captured[0].low_batch is True
        print("4. CUDA Graph DecodeStep generation verified successfully.")
        
    print("=" * 60)
    print("ALL EXECUTOR RUNTIME & GATING TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_executor_instantiation_and_gating()
