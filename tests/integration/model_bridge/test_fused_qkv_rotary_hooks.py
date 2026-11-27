"""hook_rot_q / hook_rot_k on attention bridges that apply RoPE themselves after a fused QKV.

GPT-NeoX (Pythia), Phi-3 and CodeGen rebuild attention in the bridge, so the
rotated Q and K are available, but the rotary hooks existed on these bridges
without ever firing. They must fire on the tensors that enter the attention
scores, and an edit there must reach the output. Tiny programmatic configs, no
weight downloads.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module

_SHARED = dict(vocab_size=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)

CASES = {
    # Partial rotary (rotary_pct < 1), as in the Pythia checkpoints.
    "neox": (
        "GPTNeoXForCausalLM",
        "gpt_neox",
        dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            rotary_pct=0.25,
            max_position_embeddings=64,
            **_SHARED,
        ),
    ),
    # Fused qkv_proj with grouped K/V heads.
    "phi3": (
        "Phi3ForCausalLM",
        "phi3",
        dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            **_SHARED,
        ),
    ),
    # GPT-J-style rotate_every_two on the first rotary_dim dims.
    "codegen": (
        "CodeGenForCausalLM",
        "codegen",
        dict(n_embd=64, n_layer=2, n_head=4, n_positions=64, rotary_dim=8, **_SHARED),
    ),
}


@pytest.fixture(scope="module", params=sorted(CASES), ids=sorted(CASES))
def bridge(request):
    architecture, model_type, kwargs = CASES[request.param]
    hf_config = AutoConfig.for_model(model_type, **kwargs)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    return build_bridge_from_module(model, architecture, hf_config=hf_config).eval()


@pytest.fixture(scope="module")
def tokens():
    return torch.randint(3, 128, (2, 7), generator=torch.Generator().manual_seed(1))


def test_rotary_hooks_fire_with_documented_shapes(bridge, tokens):
    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens)
    n_heads, d_head = bridge.cfg.n_heads, bridge.cfg.d_head
    n_kv_heads = getattr(bridge.cfg, "n_key_value_heads", None) or n_heads
    for layer in range(bridge.cfg.n_layers):
        assert cache[f"blocks.{layer}.attn.hook_rot_q"].shape == (2, 7, n_heads, d_head)
        assert cache[f"blocks.{layer}.attn.hook_rot_k"].shape == (2, 7, n_kv_heads, d_head)


def test_rotary_hooks_hold_the_q_and_k_behind_the_scores(bridge, tokens):
    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens)
    n_heads, d_head = bridge.cfg.n_heads, bridge.cfg.d_head
    visible = torch.tril(torch.ones(7, 7, dtype=torch.bool))
    for layer in range(bridge.cfg.n_layers):
        rot_q = cache[f"blocks.{layer}.attn.hook_rot_q"]
        rot_k = cache[f"blocks.{layer}.attn.hook_rot_k"]
        rot_k = rot_k.repeat_interleave(n_heads // rot_k.shape[2], dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", rot_q, rot_k) / d_head**0.5
        hooked = cache[f"blocks.{layer}.attn.hook_attn_scores"]
        torch.testing.assert_close(scores[..., visible], hooked[..., visible])
        # RoPE moved Q away from the projection output.
        q = cache[f"blocks.{layer}.attn.hook_q"]
        assert not torch.allclose(q, rot_q)


def test_edit_at_hook_rot_k_reaches_the_output(bridge, tokens):
    # Zero rotated keys make every visible score 0: the same as zeroing those scores.
    visible = torch.tril(torch.ones(7, 7, dtype=torch.bool))

    def zero_keys(k, hook):
        return torch.zeros_like(k)

    def zero_visible_scores(scores, hook):
        return scores.masked_fill(visible, 0.0)

    with torch.no_grad():
        clean = bridge(tokens)
        via_keys = bridge.run_with_hooks(
            tokens, fwd_hooks=[("blocks.0.attn.hook_rot_k", zero_keys)]
        )
        via_scores = bridge.run_with_hooks(
            tokens, fwd_hooks=[("blocks.0.attn.hook_attn_scores", zero_visible_scores)]
        )
    assert (via_keys - clean).abs().max() > 1e-3
    torch.testing.assert_close(via_keys, via_scores)
