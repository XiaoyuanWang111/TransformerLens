"""Integration tests for the FalconMamba architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.ssm_mixer import (
    SSMMixerBridge,
)

MODEL = "tiiuae/falcon-mamba-tiny-dev"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


def _mixers(bridge):
    return [b.mixer for b in bridge.blocks if isinstance(getattr(b, "mixer", None), SSMMixerBridge)]


class TestFalconMambaBridgeCreation:
    def test_adapter_selected(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.falcon_mamba import (
            FalconMambaArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, FalconMambaArchitectureAdapter)

    def test_mixer_carries_rms_eps(self, bridge):
        """The FalconMamba quirk: parameter-free RMS on B/C/dt, keyed on rms_eps."""
        mixers = _mixers(bridge)
        assert mixers, "no SSM mixers found"
        assert getattr(mixers[0].original_component, "rms_eps", None) is not None


class TestFalconMambaForwardEquivalence:
    def test_forward_matches_hf(self, bridge, sample_tokens):
        hf_model = bridge.original_model
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestFalconMambaEagerScan:
    def test_eager_scan_applies_bcdt_rms(self, bridge, sample_tokens):
        """The eager S6 scan must reproduce the fused path, which requires
        applying FalconMamba's B/C/dt RMS before dt_proj."""
        mixers = _mixers(bridge)
        with torch.no_grad():
            base = bridge(sample_tokens, use_cache=False)
            for m in mixers:
                m.eager_scan = True
            try:
                eager = bridge(sample_tokens, use_cache=False)
            finally:
                for m in mixers:
                    m.eager_scan = False
        rel = (eager - base).abs().max().item() / max(base.abs().max().item(), 1e-8)
        assert rel < 1e-3, f"eager vs fused rel diff {rel:.2e}"

    def test_eager_scan_applies_the_rms_once(self, bridge, sample_tokens):
        """With a large eps, RMS applied twice differs visibly from RMS applied
        once, so eager-vs-fused parity holds only if the eager scan applies
        FalconMamba's RMS once (from transformers 5.15 the mixer also holds it
        as dt/b/c_layernorm modules)."""
        mixers = _mixers(bridge)
        norms = ("dt_layernorm", "b_layernorm", "c_layernorm")
        saved = [
            (
                m.original_component.rms_eps,
                [getattr(getattr(m.original_component, n, None), "eps", None) for n in norms],
            )
            for m in mixers
        ]

        def set_eps(m, eps, module_eps):
            m.original_component.rms_eps = eps
            for n, e in zip(norms, module_eps):
                if getattr(m.original_component, n, None) is not None:
                    getattr(m.original_component, n).eps = e

        with torch.no_grad():
            for m in mixers:
                set_eps(m, 0.5, [0.5] * len(norms))
            try:
                base = bridge(sample_tokens, use_cache=False)
                for m in mixers:
                    m.eager_scan = True
                eager = bridge(sample_tokens, use_cache=False)
            finally:
                for m, (eps, module_eps) in zip(mixers, saved):
                    m.eager_scan = False
                    set_eps(m, eps, module_eps)
        rel = (eager - base).abs().max().item() / max(base.abs().max().item(), 1e-8)
        assert rel < 1e-3, f"eager vs fused rel diff {rel:.2e}"

    def test_rms_is_load_bearing(self, bridge, sample_tokens):
        """Counterfactual: hiding the RMS from the eager scan must degrade parity,
        proving the RMS branch is actually engaged for FalconMamba. The RMS is
        keyed on rms_eps, and from transformers 5.15 also held as the mixer's
        dt/b/c_layernorm modules, so both are hidden."""
        mixers = _mixers(bridge)
        norms = ("rms_eps", "dt_layernorm", "b_layernorm", "c_layernorm")
        saved = [{n: getattr(m.original_component, n, None) for n in norms} for m in mixers]
        with torch.no_grad():
            base = bridge(sample_tokens, use_cache=False)
            for m in mixers:
                m.eager_scan = True
                for n in norms:
                    if hasattr(m.original_component, n):
                        setattr(m.original_component, n, None)
            try:
                broken = bridge(sample_tokens, use_cache=False)
            finally:
                for m, values in zip(mixers, saved):
                    m.eager_scan = False
                    for n, value in values.items():
                        if value is not None:
                            setattr(m.original_component, n, value)
        rel = (broken - base).abs().max().item() / max(base.abs().max().item(), 1e-8)
        assert rel > 1e-3, f"eager scan without RMS still matched (rel {rel:.2e})"


class TestFalconMambaStateReconstruction:
    def test_ssm_state_reconstruction_runs(self, bridge, sample_tokens):
        with torch.no_grad():
            _, cache = bridge.run_with_cache(sample_tokens, use_cache=False)
        mixer = _mixers(bridge)[0]
        S = mixer.compute_ssm_state(cache, layer_idx=0)
        assert S.ndim == 4
        assert S.shape[2] == sample_tokens.shape[1]
        assert torch.isfinite(S).all()
