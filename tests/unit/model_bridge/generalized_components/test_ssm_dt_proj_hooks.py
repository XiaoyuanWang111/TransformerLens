"""dt_proj hooks of the Mamba-1 mixers fire on transformers >= 5.15.

transformers 5.15 rewrote the Mamba-1 mixers (Mamba, FalconMamba, Jamba) to compute
``self.dt_proj.weight @ time_step.transpose(1, 2)`` inline and pass the bias to the
selective scan, instead of calling ``self.dt_proj(time_step)``. The ``LinearBridge``
swapped in for ``dt_proj`` no longer ran. ``SSMMixerBridge`` now fires its hooks on
that product (``route_dt_proj_matmul``).

The first group uses a stand-in mixer shaped like transformers >= 5.15, so it checks
the routing on any installed transformers. The second wires the real HF mixers: on
transformers < 5.15 they call ``dt_proj`` themselves, on >= 5.15 the routing fires
the hooks; either way they fire with the same shapes and the output stays bit-exact
with HF's.
"""

from __future__ import annotations

import copy
import sys

import pytest
import torch
import torch.nn.functional as F

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.component_setup import setup_submodules
from transformer_lens.model_bridge.generalized_components import LinearBridge
from transformer_lens.model_bridge.generalized_components.ssm_mixer import (
    route_dt_proj_matmul,
)

RANK, CHANNELS = 3, 8


def mamba_selective_scan(dt, delta_bias=None, during=None):
    """Stand-in for transformers' scan: adds the bias, then the softplus."""
    if during is not None:
        during()
    if delta_bias is not None:
        dt = dt + delta_bias.to(dt.dtype)[..., None]
    return F.softplus(dt)


class _InlineDtMixer(torch.nn.Module):
    """Computes dt as transformers >= 5.15 does: inline product, bias added by the scan."""

    def __init__(self) -> None:
        super().__init__()
        self.dt_proj = torch.nn.Linear(RANK, CHANNELS)
        self.other = torch.nn.Linear(RANK, CHANNELS)

    def forward(self, time_step: torch.Tensor, use_other: bool = False, during=None):
        proj = self.other if use_other else self.dt_proj
        dt = proj.weight @ time_step.transpose(1, 2)  # [batch, channels, seq]
        return mamba_selective_scan(dt, proj.bias.float(), during=during)


def _bridged_inline_mixer():
    torch.manual_seed(0)
    mixer = _InlineDtMixer()
    reference = copy.deepcopy(mixer)
    dt_proj = LinearBridge(name="dt_proj")
    dt_proj.set_original_component(mixer.dt_proj)
    mixer.dt_proj = dt_proj
    return mixer, reference, dt_proj


def _record(store: dict, name: str):
    def hook(tensor, hook):
        store[name] = tensor.detach().clone()
        return tensor

    return hook


def test_route_fires_dt_proj_hooks_and_keeps_the_output_bit_exact() -> None:
    mixer, reference, dt_proj = _bridged_inline_mixer()
    seen: dict = {}
    dt_proj.hook_in.add_hook(_record(seen, "in"))
    dt_proj.hook_out.add_hook(_record(seen, "out"))
    time_step = torch.randn(2, 5, RANK)
    with torch.no_grad(), route_dt_proj_matmul(mixer, "dt_proj"):
        out = mixer(time_step)
    torch.testing.assert_close(out, reference(time_step), rtol=0.0, atol=0.0)
    torch.testing.assert_close(seen["in"], time_step, rtol=0.0, atol=0.0)
    # hook_out holds what dt_proj(time_step) returns.
    torch.testing.assert_close(seen["out"], reference.dt_proj(time_step))


def test_route_patch_on_dt_proj_hook_out_reaches_the_output() -> None:
    mixer, _, dt_proj = _bridged_inline_mixer()
    dt_proj.hook_out.add_hook(lambda tensor, hook: torch.full_like(tensor, 0.5))
    with torch.no_grad(), route_dt_proj_matmul(mixer, "dt_proj"):
        out = mixer(torch.randn(2, 5, RANK))
    torch.testing.assert_close(out, F.softplus(torch.full_like(out, 0.5)))


def test_route_in_place_patch_on_dt_proj_hook_out_reaches_the_output() -> None:
    mixer, _, dt_proj = _bridged_inline_mixer()

    def zero_in_place(tensor, hook):
        tensor.zero_()

    dt_proj.hook_out.add_hook(zero_in_place)
    with torch.no_grad(), route_dt_proj_matmul(mixer, "dt_proj"):
        out = mixer(torch.randn(2, 5, RANK))
    torch.testing.assert_close(out, F.softplus(torch.zeros_like(out)))


def test_route_patch_on_dt_proj_hook_in_reaches_the_output() -> None:
    mixer, reference, dt_proj = _bridged_inline_mixer()
    dt_proj.hook_in.add_hook(lambda tensor, hook: torch.zeros_like(tensor))
    with torch.no_grad(), route_dt_proj_matmul(mixer, "dt_proj"):
        out = mixer(torch.randn(2, 5, RANK))
    torch.testing.assert_close(out, reference(torch.zeros(2, 5, RANK)), rtol=0.0, atol=0.0)


def test_route_leaves_other_matmuls_alone() -> None:
    mixer, reference, dt_proj = _bridged_inline_mixer()
    seen: dict = {}
    dt_proj.hook_in.add_hook(_record(seen, "in"))
    time_step = torch.randn(2, 5, RANK)
    with torch.no_grad(), route_dt_proj_matmul(mixer, "dt_proj"):
        out = mixer(time_step, use_other=True)
    torch.testing.assert_close(out, reference(time_step, use_other=True), rtol=0.0, atol=0.0)
    assert seen == {}


def test_route_ends_when_the_scan_starts_and_restores_the_scan() -> None:
    mixer, _, dt_proj = _bridged_inline_mixer()
    calls = []
    dt_proj.hook_in.add_hook(lambda tensor, hook: calls.append(hook.name))
    weight = dt_proj.original_component.weight
    product_in_scan = lambda: weight @ torch.randn(2, RANK, 5)  # noqa: E731
    with torch.no_grad():
        with route_dt_proj_matmul(mixer, "dt_proj"):
            mixer(torch.randn(2, 5, RANK), during=product_in_scan)
        product_in_scan()
    assert len(calls) == 1
    assert sys.modules[__name__].mamba_selective_scan is mamba_selective_scan


def test_route_restores_the_scan_and_leaves_the_mode_after_an_error() -> None:
    mixer, _, dt_proj = _bridged_inline_mixer()
    calls = []

    def fail(tensor, hook):
        calls.append(hook.name)
        raise RuntimeError("hook failed")

    dt_proj.hook_in.add_hook(fail)
    with pytest.raises(RuntimeError, match="hook failed"):
        with torch.no_grad(), route_dt_proj_matmul(mixer, "dt_proj"):
            mixer(torch.randn(2, 5, RANK))
    dt_proj.original_component.weight @ torch.randn(2, RANK, 5)
    assert len(calls) == 1
    assert sys.modules[__name__].mamba_selective_scan is mamba_selective_scan


def test_route_is_a_no_op_without_a_dt_proj_bridge() -> None:
    torch.manual_seed(0)
    mixer = _InlineDtMixer()
    time_step = torch.randn(2, 5, RANK)
    with torch.no_grad():
        expected = mixer(time_step)
        with route_dt_proj_matmul(mixer, "dt_proj"):
            torch.testing.assert_close(mixer(time_step), expected, rtol=0.0, atol=0.0)


# --- Real HF mixers, wired as the adapters wire them --------------------------------


def _wire(arch: str, hf_module):
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(
        make_bridge_cfg(arch, d_model=32, n_heads=4, d_head=8)
    )
    bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules["mixer"])
    bridge.set_original_component(hf_module)
    setup_submodules(bridge, adapter, hf_module)
    return bridge


def _mamba():
    from transformers import MambaConfig
    from transformers.models.mamba.modeling_mamba import MambaMixer

    cfg = MambaConfig(
        hidden_size=32, state_size=8, expand=2, conv_kernel=4, num_hidden_layers=2, vocab_size=64
    )
    return MambaMixer(cfg, layer_idx=0)


def _falcon_mamba():
    from transformers import FalconMambaConfig, FalconMambaForCausalLM

    cfg = FalconMambaConfig(
        hidden_size=32, state_size=8, expand=2, conv_kernel=4, num_hidden_layers=2, vocab_size=64
    )
    # Built through the model: FalconMambaMixer(cfg, 0) alone fails in its weight init.
    return FalconMambaForCausalLM(cfg).backbone.layers[0].mixer


def _jamba():
    from transformers import JambaConfig
    from transformers.models.jamba.modeling_jamba import JambaMambaMixer

    cfg = JambaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=4,
        use_mamba_kernels=False,
    )
    return JambaMambaMixer(cfg, layer_idx=0)


HF_MIXERS = [
    pytest.param("MambaForCausalLM", _mamba, id="mamba"),
    pytest.param("FalconMambaForCausalLM", _falcon_mamba, id="falcon_mamba"),
    pytest.param("JambaForCausalLM", _jamba, id="jamba"),
]


@pytest.mark.parametrize("arch,build", HF_MIXERS)
def test_hf_mixer_dt_proj_hooks_fire_and_output_is_bit_exact(arch, build) -> None:
    torch.manual_seed(0)
    hf_module = build().eval()
    reference = copy.deepcopy(hf_module)
    bridge = _wire(arch, hf_module)
    seen: dict = {}
    bridge.dt_proj.hook_in.add_hook(_record(seen, "in"))
    bridge.dt_proj.hook_out.add_hook(_record(seen, "out"))

    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        out = bridge(x)
        expected = reference(x)

    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
    linear = bridge.dt_proj.original_component
    assert seen["in"].shape == (2, 7, linear.in_features)
    assert seen["out"].shape == (2, 7, linear.out_features)
    torch.testing.assert_close(seen["out"], F.linear(seen["in"], linear.weight, linear.bias))


@pytest.mark.parametrize("arch,build", HF_MIXERS)
def test_hf_mixer_patch_on_dt_proj_hook_out_reaches_the_output(arch, build) -> None:
    torch.manual_seed(0)
    hf_module = build().eval()
    reference = copy.deepcopy(hf_module)
    bridge = _wire(arch, hf_module)
    bridge.dt_proj.hook_out.add_hook(lambda tensor, hook: tensor - 5.0)

    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        assert not torch.allclose(bridge(x), reference(x))
