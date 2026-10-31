"""Infrastructure for the evaluation suite. CPU, tiny config, no checkpoint."""

from __future__ import annotations

import pytest
import torch

from ffnmem.analysis_mech import linear_cka, mcnemar_p
from ffnmem.config import load_config, resolve_model
from ffnmem.hooks import Intervention, memory_means
from ffnmem.inference.decode import decode_all
from ffnmem.inference.offload import offload_tables
from ffnmem.model import Transformer
from ffnmem.scoring import batch_nll, greedy_generate, pack_request, rolling_windows, score_packed
from ffnmem.trainable import apply_trainable
from analysis.perf_model import n_star, overlap_us


def _tiny(delay: int = 0, rank1: int = 0):
    cfg = load_config("configs/tiny_cpu.yaml")
    if delay:
        cfg["model"]["memory_delay"] = delay
    if rank1:
        cfg["model"]["memory_overrides"] = {1: {"rank1": {"dim": rank1}}}
    if delay or rank1:
        cfg["model"] = resolve_model(cfg["model"])
    model = Transformer(cfg["model"])
    model.set_attention_backend("naive")
    model.eval()
    return model


def test_batch_nll_matches_manual_loss():
    model = _tiny()
    ids = torch.randint(4, model.cfg["vocab"], (2, 12))
    _, loss = model(ids[:, :-1], ids[:, 1:])
    assert torch.allclose(batch_nll(model, ids), loss, atol=1e-5)


def test_trainable_regimes_freeze_the_right_params():
    model = _tiny()
    info = apply_trainable(model, "memory")
    for name, param in model.named_parameters():
        assert param.requires_grad == (".memory." in name)
    assert info["trainable"] > 0
    model = _tiny()
    apply_trainable(model, "backbone")
    for name, param in model.named_parameters():
        assert param.requires_grad == (".memory." not in name)
    model = _tiny()
    apply_trainable(model, "tables")
    for name, param in model.named_parameters():
        want = ".memory.module.tables." in name or ".memory.rank1_tables." in name
        assert param.requires_grad == want
    model = _tiny()
    apply_trainable(model, "ffn_layers:0,1")
    for name, param in model.named_parameters():
        want = name.startswith("blocks.0.ffn.") or name.startswith("blocks.1.ffn.")
        assert param.requires_grad == want
    model = _tiny()
    apply_trainable(model, "all")
    assert all(p.requires_grad for p in model.parameters())


def test_zero_intervention_matches_a_manual_zero():
    model = _tiny()
    ids = torch.randint(4, model.cfg["vocab"], (1, 8))
    with Intervention(model, memory="zero"):
        zeroed = model(ids)
    # A second, independent hook must agree.
    saved = model.blocks[1].memory.forward

    def manual(h, h_norm, _orig=saved):
        return torch.zeros_like(_orig(h, h_norm))

    model.blocks[1].memory.forward = manual
    try:
        again = model(ids)
    finally:
        model.blocks[1].memory.forward = saved
    assert torch.allclose(zeroed, again, atol=1e-5)


def test_patch_with_clean_cache_reproduces_clean_logits():
    model = _tiny()
    ids = torch.randint(4, model.cfg["vocab"], (1, 8))
    clean = model(ids)
    store = {}
    with Intervention(model, memory="capture", store=store):
        model(ids)
    patch = {pos: tensors[0] for pos, tensors in store["memory"].items()}
    with Intervention(model, memory="patch", patch={"memory": patch}):
        patched = model(ids)
    assert torch.allclose(patched, clean, atol=1e-5)


def test_decode_matches_full_forward():
    model = _tiny()
    ids = torch.randint(4, model.cfg["vocab"], (1, 6))
    full = model(ids)
    inc = decode_all(model, ids)
    assert torch.allclose(inc, full, atol=1e-4, rtol=1e-4)


def test_delayed_decode_matches_full_forward():
    model = _tiny(delay=1)
    assert model.memory_injection_map[1] == 2
    ids = torch.randint(4, model.cfg["vocab"], (1, 6))
    full = model(ids)
    inc = decode_all(model, ids)
    assert torch.allclose(inc, full, atol=1e-4, rtol=1e-4)


def test_offload_sync_matches_gpu_mode():
    model = _tiny()
    ids = torch.randint(4, model.cfg["vocab"], (1, 6))
    base = model(ids)
    with offload_tables(model, "sync"):
        off = model(ids)
    assert torch.allclose(off, base, atol=1e-5)
    # restored
    assert torch.allclose(model(ids), base, atol=1e-5)


def test_batched_scoring_matches_one_at_a_time():
    model = _tiny()
    g = torch.Generator().manual_seed(0)
    packed = []
    for n_ctx, n_cont in [(3, 2), (9, 1), (5, 4), (1, 3)]:
        ctx = torch.randint(4, model.cfg["vocab"], (n_ctx,), generator=g).tolist()
        cont = torch.randint(4, model.cfg["vocab"], (n_cont,), generator=g).tolist()
        packed.append(pack_request(ctx, cont, model.cfg["seq_len"]))
    batched = score_packed(model, packed, torch.device("cpu"), token_budget=1000, fp16=False)
    single = [score_packed(model, [p], torch.device("cpu"), fp16=False)[0] for p in packed]
    for (a, ga), (b, gb) in zip(batched, single):
        assert abs(a - b) < 1e-4 and ga == gb


def test_rolling_windows_score_every_token_once():
    ids = list(range(10, 47))
    wins = rolling_windows(ids, seq_len=8)
    scored = []
    for win, n in wins:
        assert len(win) <= 9
        scored += win[-n:]
    assert scored == ids


@pytest.mark.parametrize("rank1", [0, 8])
def test_cached_generation_matches_reforward(rank1):
    model = _tiny(rank1=rank1)
    if rank1:
        assert model.blocks[1].memory.rank1_dim == rank1
        torch.nn.init.normal_(model.blocks[1].memory.rank1_out.weight, std=0.5)
    prompt = torch.randint(4, model.cfg["vocab"], (9,)).tolist()
    a = greedy_generate(model, prompt, 6, torch.device("cpu"), model.cfg["seq_len"], use_cache=True)
    b = greedy_generate(model, prompt, 6, torch.device("cpu"), model.cfg["seq_len"], use_cache=False)
    assert a == b


def test_mean_intervention_uses_the_mean_vector():
    model = _tiny()
    ids = torch.randint(4, model.cfg["vocab"], (2, 8))
    means = memory_means(model, [ids], torch.device("cpu"))
    assert set(means) == {i for i, b in enumerate(model.blocks) if b.memory is not None}
    store = {}
    with Intervention(model, memory="capture", store=store):
        model(ids)
    pos = next(iter(means))
    assert torch.allclose(means[pos], store["memory"][pos][0].float().mean(dim=(0, 1)), atol=1e-5)
    with Intervention(model, memory="mean", mean=means):
        out = model(ids)
    assert out.shape == model(ids).shape


def test_cka_of_a_matrix_with_itself_is_one():
    x = torch.randn(8, 5)
    assert abs(linear_cka(x, x) - 1.0) < 1e-4


def test_mcnemar_is_one_when_models_agree():
    assert mcnemar_p([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_n_star_is_the_smallest_covering_delay():
    layers = [10.0] * 6
    # overlap(n) = 10 * n  (layers 1..n)
    assert overlap_us(2, layers) == 20.0
    assert n_star(layers, 15.0) == 2
    assert n_star(layers, 0.0) == 0
