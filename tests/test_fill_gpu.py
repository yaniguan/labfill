from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from labfill import fill_gpu as fill_gpu_mod
from labfill.fill_gpu import (
    CudaGraphStepRunner,
    PackCollate,
    ProtocolTransformer,
    SlowTokenizerDataset,
    _sequence_start_offsets,
    build_batches_from_items,
    build_block_diag_mask,
    build_cu_seqlens,
    build_flex_block_mask,
    chunk_sequence_pairs,
    collate_padded_pairs,
    collate_varlen_pairs,
    compute_packed_loss,
    count_batch_training_flops,
    detect_gpu_peak_flops,
    flex_local_global_attention,
    generate_protocol_sequences,
    pack_lengths_obfd,
    pack_sequence_pairs,
    pack_sequences_obfd,
    packed_sequence_ce_loss,
    probe_flex_attention,
    run_fill_gpu_ablation,
    run_torch_profiler_summary,
    tiled_local_global_attention,
)


def _seq_id_row(lengths: list[int], s: int) -> torch.Tensor:
    sid = torch.zeros((1, s), dtype=torch.int64)
    start = 0
    for k, n in enumerate(lengths, start=1):
        sid[0, start : start + n] = k
        start += n
    return sid


def _reference_local_global_mask(seq_id: torch.Tensor, w: int, g: int, causal: bool) -> torch.Tensor:
    """Dense (B, 1, S, S) mask with per-sequence G globals + [prev_blk, cur_blk], each key counted once."""
    b, s = seq_id.shape
    seq_starts = _sequence_start_offsets(seq_id)  # (B, S)
    i = torch.arange(s).view(1, s, 1)
    j = torch.arange(s).view(1, 1, s)
    in_local = ((j // w) == (i // w)) | ((j // w) == (i // w) - 1)
    is_seq_glob = (j - seq_starts.unsqueeze(1)) < g
    in_window = in_local | is_seq_glob
    if causal:
        in_window = in_window & (j <= i)
    same = (seq_id.unsqueeze(2) == seq_id.unsqueeze(1)) & (seq_id.unsqueeze(2) > 0)
    allowed = same & in_window
    eye = torch.eye(s, dtype=torch.bool).unsqueeze(0).expand(b, s, s)
    return torch.where((seq_id == 0).unsqueeze(2), eye, allowed).unsqueeze(1)


@pytest.mark.parametrize(("length", "causal"), [(40, False), (64, False), (128, True)])
def test_block_sparse_matches_dense_when_sequence_fits_window(length: int, causal: bool) -> None:
    # Non-causal block-0 queries only see block 0, so the sequence must fit in W=64 tokens;
    # causal queries never look ahead, so a 2W = 128 token sequence is fully covered.
    torch.manual_seed(0)
    s, w = 256, 64
    q, k, v = (torch.randn(2, 4, s, 16) for _ in range(3))
    sid = _seq_id_row([length], s).expand(2, s).contiguous()

    sparse = tiled_local_global_attention(q, k, v, seq_id=sid, block_size=w, num_global_tokens=4, causal=causal)
    dense = F.scaled_dot_product_attention(q, k, v, attn_mask=build_block_diag_mask(sid, causal=causal))
    assert torch.allclose(sparse[:, :, :length], dense[:, :, :length], atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_block_sparse_matches_dense_with_same_sparsity_mask(causal: bool) -> None:
    torch.manual_seed(1)
    s, w, g = 512, 64, 4
    q, k, v = (torch.randn(1, 2, s, 8) for _ in range(3))
    sid = _seq_id_row([150, 3, 200, 90, 40], s)

    sparse = tiled_local_global_attention(q, k, v, seq_id=sid, block_size=w, num_global_tokens=g, causal=causal)
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=_reference_local_global_mask(sid, w=w, g=g, causal=causal)
    )
    assert torch.allclose(sparse, ref, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graphs need a CUDA device")
def test_cuda_graph_runner_captures_optimizer_step() -> None:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, capturable=True)
    seqs = [
        (torch.randint(4, 200, (n,), dtype=torch.int64), torch.randint(0, 8, (n,), dtype=torch.int64))
        for n in (30, 50, 70)
    ]
    batch = pack_sequence_pairs(seqs, pack_len=128, rows_per_batch=2, attn_mode="dense")[0]
    runner = CudaGraphStepRunner(model, opt, pack_rows=batch.input_ids.shape[0], pack_len=128, device=dev)

    before = [p.detach().clone() for p in model.parameters()]
    loss1 = runner.replay_step(batch.to(dev))
    loss2 = runner.replay_step(batch.to(dev))
    torch.cuda.synchronize()
    assert torch.isfinite(torch.tensor([loss1, loss2])).all()
    assert any(not torch.equal(b, p) for b, p in zip(before, list(model.parameters()), strict=True))


@pytest.mark.parametrize("causal", [False, True])
def test_sequence_packing_and_varlen_no_cross_contamination(causal: bool) -> None:
    torch.manual_seed(0)
    model = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, causal=causal)
    model.eval()

    seq_a = (torch.tensor([1, 17, 33, 49, 65, 2], dtype=torch.int64), torch.zeros(6, dtype=torch.int64))
    seq_b1 = (torch.tensor([1, 18, 34, 50, 66, 70, 80, 2], dtype=torch.int64), torch.zeros(8, dtype=torch.int64))
    seq_b2 = (torch.tensor([1, 200, 201, 202, 203, 204, 205, 2], dtype=torch.int64), torch.zeros(8, dtype=torch.int64))

    # 1. Dense packed mode: mutating seq_b tokens must NOT alter seq_a logits
    pk1 = pack_sequence_pairs([seq_a, seq_b1], pack_len=128, rows_per_batch=1, attn_mode="dense")[0]
    pk2 = pack_sequence_pairs([seq_a, seq_b2], pack_len=128, rows_per_batch=1, attn_mode="dense")[0]
    assert pk1.position_ids[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert pk1.position_ids[0, 6:14].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with torch.no_grad():
            out1 = model(pk1.input_ids, pk1.position_ids, seq_id=pk1.seq_id, mode="dense")
            out2 = model(pk2.input_ids, pk2.position_ids, seq_id=pk2.seq_id, mode="dense")
            assert torch.allclose(out1[0, :6], out2[0, :6], atol=1e-5)

            # 2. Tiled block-sparse mode on 256-token row: mutating seq_b must NOT alter seq_a
            pks1 = pack_sequence_pairs([seq_a, seq_b1], pack_len=256, rows_per_batch=1, attn_mode="block_sparse")[0]
            pks2 = pack_sequence_pairs([seq_a, seq_b2], pack_len=256, rows_per_batch=1, attn_mode="block_sparse")[0]
            outs1 = model(pks1.input_ids, pks1.position_ids, seq_id=pks1.seq_id, mode="block_sparse")
            outs2 = model(pks2.input_ids, pks2.position_ids, seq_id=pks2.seq_id, mode="block_sparse")
            assert torch.allclose(outs1[0, :6], outs2[0, :6], atol=1e-5)

            # 3. Jagged varlen mode matches standalone single-sequence forward pass with zero UserWarnings
            vl = collate_varlen_pairs([seq_a, seq_b1])
            out_vl = model(vl.input_ids, vl.position_ids, cu_seqlens=vl.cu_seqlens, mode="varlen")
            single_a = collate_padded_pairs([seq_a])
            out_single = model(single_a.input_ids, single_a.position_ids, seq_id=single_a.seq_id, mode="dense")
            assert torch.allclose(out_vl[:6], out_single[0, :6], atol=1e-5)

            # 4. When causal=True, mutating future tokens within seq_b must NOT alter earlier prefix logits of seq_b
            if causal:
                seq_b_fut = (
                    torch.tensor([1, 18, 34, 50, 199, 200, 201, 2], dtype=torch.int64),
                    torch.zeros(8, dtype=torch.int64),
                )
                pk_fut = pack_sequence_pairs([seq_a, seq_b_fut], pack_len=256, rows_per_batch=1, attn_mode="dense")[0]
                out_fut_d = model(pk_fut.input_ids, pk_fut.position_ids, seq_id=pk_fut.seq_id, mode="dense")
                out_orig_d = model(pks1.input_ids, pks1.position_ids, seq_id=pks1.seq_id, mode="dense")
                assert torch.allclose(out_fut_d[0, 6:10], out_orig_d[0, 6:10], atol=1e-5)

                out_fut_s = model(pk_fut.input_ids, pk_fut.position_ids, seq_id=pk_fut.seq_id, mode="block_sparse")
                assert torch.allclose(out_fut_s[0, 6:10], outs1[0, 6:10], atol=1e-5)


def test_block_sparse_packing_order_invariance_for_long_sequences() -> None:
    torch.manual_seed(7)
    model = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, causal=True, block_size=64)
    model.eval()

    # Block-aligned short (64 = W) and long (192 = 3W > 2W) sequences packed in both orders into S=256
    seq_short = (torch.randint(4, 200, (64,), dtype=torch.int64), torch.zeros(64, dtype=torch.int64))
    seq_long = (torch.randint(4, 200, (192,), dtype=torch.int64), torch.zeros(192, dtype=torch.int64))

    b_first = pack_sequence_pairs([seq_long, seq_short], pack_len=256, rows_per_batch=1, attn_mode="block_sparse")[0]
    b_second = pack_sequence_pairs([seq_short, seq_long], pack_len=256, rows_per_batch=1, attn_mode="block_sparse")[0]

    with torch.no_grad():
        out_first = model(b_first.input_ids, b_first.position_ids, seq_id=b_first.seq_id, mode="block_sparse")
        out_second = model(b_second.input_ids, b_second.position_ids, seq_id=b_second.seq_id, mode="block_sparse")

    assert torch.allclose(out_first[0, :192], out_second[0, 64:256], atol=1e-5)
    assert torch.allclose(out_first[0, 192:256], out_second[0, :64], atol=1e-5)


def test_pack_sequence_pairs_truncation_guard_and_pack_collate() -> None:
    over = [(torch.ones(60, dtype=torch.int64), torch.zeros(60, dtype=torch.int64))]
    with pytest.raises(ValueError, match="pack_len=32"):
        pack_sequence_pairs(over, pack_len=32)

    trunc_b = pack_sequence_pairs(over, pack_len=32, truncate=True)[0]
    assert trunc_b.real_tokens == 32

    items = generate_protocol_sequences(num_sequences=12, min_len=8, max_len=128, seed=3)
    ds = SlowTokenizerDataset(items, slow_work_iters=2)
    loader = DataLoader(ds, batch_size=6, shuffle=False, num_workers=0, collate_fn=PackCollate(pack_len=192))
    batches = list(loader)
    assert all(b.input_ids.shape[1] == 192 for b in batches)
    assert sum(b.real_tokens for b in batches) == sum(len(it.tokens) for it in items)


def test_analytic_flop_accounting() -> None:
    model = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=256, causal=False)
    model_causal = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=256, causal=True)
    short = (torch.ones(10, dtype=torch.int64), torch.zeros(10, dtype=torch.int64))
    long = (torch.ones(100, dtype=torch.int64), torch.zeros(100, dtype=torch.int64))

    padded = collate_padded_pairs([short, long])
    u_pad, e_pad = count_batch_training_flops(model, padded)
    assert 0.0 < u_pad < e_pad

    # Causal model halves quadratic attention pairs: 0.5 * s * (s + 1) < s * s
    u_causal, _ = count_batch_training_flops(model_causal, padded)
    expected_diff = 3.0 * float(4 * model.n_layers * model.d_model) * (
        (10 * 10 + 100 * 100) - (0.5 * 10 * 11 + 0.5 * 100 * 101)
    )
    assert abs((u_pad - u_causal) - expected_diff) < 1e-4

    vl = collate_varlen_pairs([short, long])
    u_vl, e_vl = count_batch_training_flops(model, vl)
    assert abs(u_vl - e_vl) < 1e-6
    assert abs(u_vl - u_pad) < 1e-6

    # Non-multiple pack_len=150 falls back to dense SDPA in block_sparse mode -> executed FLOPs match dense!
    sp_fallback = pack_sequence_pairs([short, long], pack_len=150, rows_per_batch=1, attn_mode="block_sparse")[0]
    dn_150 = pack_sequence_pairs([short, long], pack_len=150, rows_per_batch=1, attn_mode="dense")[0]
    _, e_sp150 = count_batch_training_flops(model, sp_fallback)
    _, e_dn150 = count_batch_training_flops(model, dn_150)
    assert abs(e_sp150 - e_dn150) < 1e-6

    # Multiple pack_len=256 (> 2 * 64) uses sparse 2W + G window -> executed FLOPs strictly below dense!
    sp_256 = pack_sequence_pairs([short, long], pack_len=256, rows_per_batch=1, attn_mode="block_sparse")[0]
    dn_256 = pack_sequence_pairs([short, long], pack_len=256, rows_per_batch=1, attn_mode="dense")[0]
    _, e_sp256 = count_batch_training_flops(model, sp_256)
    _, e_dn256 = count_batch_training_flops(model, dn_256)
    assert e_sp256 < e_dn256


def test_flex_attention_execution_path_matches_tiled_sdpa() -> None:
    info = probe_flex_attention("cpu")
    assert info["forward_ok"] is True

    torch.manual_seed(11)
    s, w, g = 256, 64, 4
    q, k, v = (torch.randn(1, 4, s, 16) for _ in range(3))
    sid = _seq_id_row([64, 150, 20], s)

    with torch.no_grad():
        out_tiled = tiled_local_global_attention(q, k, v, seq_id=sid, block_size=w, num_global_tokens=g, causal=True)
        out_flex = flex_local_global_attention(q, k, v, seq_id=sid, block_size=w, num_global_tokens=g, causal=True)
    assert torch.allclose(out_tiled, out_flex, atol=1e-4)

    torch.manual_seed(12)
    m_tiled = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, causal=False, use_flex_attention=False)
    torch.manual_seed(12)
    m_flex = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, causal=False, use_flex_attention=True)
    m_tiled.eval()
    m_flex.eval()
    pk = pack_sequence_pairs(
        [(torch.randint(4, 100, (80,), dtype=torch.int64), torch.zeros(80, dtype=torch.int64))],
        pack_len=256,
        rows_per_batch=1,
        attn_mode="block_sparse",
    )[0]
    with torch.no_grad():
        logits_t = m_tiled(pk.input_ids, pk.position_ids, seq_id=pk.seq_id, mode="block_sparse")
        logits_f = m_flex(pk.input_ids, pk.position_ids, seq_id=pk.seq_id, mode="block_sparse")
    assert torch.allclose(logits_t[0, :80], logits_f[0, :80], atol=1e-4)


def test_cuda_graph_runner_static_seq_id_reset_without_cuda() -> None:
    # Verify replay_step resets static_seq_id to 1s when batch.seq_id is None
    runner = object.__new__(CudaGraphStepRunner)
    runner.static_input_ids = torch.zeros((1, 8), dtype=torch.int64)
    runner.static_pos_ids = torch.zeros((1, 8), dtype=torch.int64)
    runner.static_seq_id = torch.tensor([[1, 1, 2, 2, 0, 0, 0, 0]], dtype=torch.int64)
    runner.static_labels = torch.zeros((1, 8), dtype=torch.int64)
    runner.static_loss = torch.tensor(0.5)
    runner._captured = True

    class _DummyGraph:
        def replay(self) -> None:
            return None

    runner.graph = _DummyGraph()  # type: ignore[assignment]
    b_no_sid = collate_padded_pairs([(torch.ones(8, dtype=torch.int64), torch.zeros(8, dtype=torch.int64))])
    b_no_sid.seq_id = None
    loss = runner.replay_step(b_no_sid)
    assert loss == 0.5
    assert runner.static_seq_id.tolist() == [[1] * 8]

    # Small dataset with pack_rows_per_batch=10 (so full_packed is empty) still returns all 10 rows
    rows_small, _, _ = run_fill_gpu_ablation(
        num_sequences=8,
        batch_size=4,
        pack_len=256,
        pack_rows_per_batch=10,
        d_model=32,
        n_layers=1,
        n_heads=2,
        slow_work_iters=5,
        seed=0,
    )
    assert len(rows_small) == 10
    assert rows_small[-1].config == "i_cuda_graphs"


def test_ablation_harness_end_to_end_tiny() -> None:
    rows, prof_ops, flex_info = run_fill_gpu_ablation(
        num_sequences=24,
        batch_size=8,
        pack_len=256,
        pack_rows_per_batch=1,
        d_model=64,
        n_layers=2,
        n_heads=4,
        slow_work_iters=20,
        peak_flops=None,
        seed=0,
    )
    assert len(rows) == 10
    by_name = {r.config: r for r in rows}
    assert by_name["a_padded_baseline"].status == "ok"
    assert by_name["b_length_bucketed"].useful_token_fraction > by_name["a_padded_baseline"].useful_token_fraction
    assert by_name["d_varlen_nested_sdpa"].useful_token_fraction == 1.0
    assert by_name["e_packed_bf16_autocast"].status == "ok"
    assert by_name["g2_async_dataloader"].status == "ok"
    assert by_name["h_block_sparse_bf16"].status == "ok"
    if not flex_info["backward_ok"]:
        assert "Tiled SDPA" in by_name["h_block_sparse_bf16"].notes
    assert by_name["a_padded_baseline"].mfu == "N/A"

    # Every active config (a, b, c, d, e, g1, g2, h) must process the exact same real token count
    expected_real = by_name["a_padded_baseline"].real_tokens
    assert expected_real > 0
    for cfg in (
        "a_padded_baseline",
        "b_length_bucketed",
        "c_sequence_packed",
        "d_varlen_nested_sdpa",
        "e_packed_bf16_autocast",
        "g1_sync_slow_tokenizer",
        "g2_async_dataloader",
        "h_block_sparse_bf16",
    ):
        assert by_name[cfg].real_tokens == expected_real, cfg

    # Verify build_batches_from_items respects pack_len=256
    items_check = generate_protocol_sequences(num_sequences=24, min_len=8, max_len=256, seed=0)
    pk_check = build_batches_from_items(items_check, "packed", pack_len=256, pack_rows_per_batch=1)
    assert all(b.input_ids.shape[1] == 256 for b in pk_check)

    assert len(prof_ops) > 0
    assert "forward_ok" in flex_info


def test_build_cu_seqlens_from_lengths_and_multi_row_seq_id() -> None:
    cu_len, max_l = build_cu_seqlens([40, 95, 25], dtype=torch.int32)
    assert cu_len.dtype == torch.int32
    assert cu_len.tolist() == [0, 40, 135, 160]
    assert max_l == 95

    # Multi-row seq_id where row 0 ends at pad (or sid=2) and row 1 starts again at sid=1
    sid = torch.tensor(
        [
            [1, 1, 1, 2, 2, 0, 0, 0],
            [1, 1, 1, 1, 2, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    cu_sid, max_sid = build_cu_seqlens(sid)
    assert cu_sid.tolist() == [0, 3, 5, 9, 10]
    assert max_sid == 4

    empty_cu, empty_max = build_cu_seqlens(torch.zeros((2, 16), dtype=torch.int64))
    assert empty_cu.tolist() == [0]
    assert empty_max == 0


@pytest.mark.parametrize("causal", [False, True])
def test_packed_varlen_matches_dense_fp32_and_bf16_at_pack_len_4096(
    causal: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(21)
    model = ProtocolTransformer(
        max_seq_len=4096,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        causal=causal,
    ).eval()

    pairs = [
        (torch.randint(4, 220, (ln,), dtype=torch.int64), torch.randint(0, 8, (ln,), dtype=torch.int64))
        for ln in (64, 180, 96, 120)
    ]
    dense_4096 = pack_sequence_pairs(pairs, pack_len=4096, rows_per_batch=1, attn_mode="dense")[0]
    pvar_4096 = pack_sequence_pairs(pairs, pack_len=4096, rows_per_batch=1, attn_mode="packed_varlen")[0]
    dense_ref = pack_sequence_pairs(pairs, pack_len=512, rows_per_batch=1, attn_mode="dense")[0]

    assert pvar_4096.input_ids.ndim == 1
    assert pvar_4096.input_ids.numel() == sum(ln for ln, _ in ( (int(t.numel()), 0) for t, _ in pairs ))
    assert pvar_4096.cu_seqlens is not None
    assert int(pvar_4096.cu_seqlens[-1].item()) == pvar_4096.real_tokens
    assert pvar_4096.max_seqlen == 180

    # Compute reference output with explicit dense mask at S=512 (< 4096)
    with torch.no_grad():
        logits_ref = model(dense_ref.input_ids, dense_ref.position_ids, seq_id=dense_ref.seq_id, mode="dense")

    # At pack_len == 4096, neither mode='packed_varlen' nor mode='dense' may allocate build_block_diag_mask!
    def _forbid_dense_mask(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("build_block_diag_mask should not be called when S >= 4096")

    monkeypatch.setattr(fill_gpu_mod, "build_block_diag_mask", _forbid_dense_mask)

    with torch.no_grad():
        logits_pvar = model(
            pvar_4096.input_ids,
            pvar_4096.position_ids,
            cu_seqlens=pvar_4096.cu_seqlens,
            mode="packed_varlen",
            max_seqlen=pvar_4096.max_seqlen,
        )
        logits_d4096 = model(
            dense_4096.input_ids,
            dense_4096.position_ids,
            seq_id=dense_4096.seq_id,
            mode="dense",
        )
        real_n = pvar_4096.real_tokens
        assert torch.allclose(logits_pvar, logits_ref[0, :real_n], atol=1e-4)
        assert torch.allclose(logits_d4096[0, :real_n], logits_ref[0, :real_n], atol=1e-4)

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            bf16_pvar = model(
                pvar_4096.input_ids,
                pvar_4096.position_ids,
                cu_seqlens=pvar_4096.cu_seqlens,
                mode="packed_varlen",
                max_seqlen=pvar_4096.max_seqlen,
            )
            bf16_ref = model(
                dense_4096.input_ids,
                dense_4096.position_ids,
                seq_id=dense_4096.seq_id,
                mode="dense",
            )
        assert torch.allclose(bf16_pvar.float(), bf16_ref[0, :real_n].float(), atol=1e-2)

    u_pv, e_pv = count_batch_training_flops(model, pvar_4096)
    assert abs(u_pv - e_pv) < 1e-6


def test_packed_sequence_ce_loss_matches_unpacked_per_sequence_mean_gradients() -> None:
    torch.manual_seed(31)
    short = (
        torch.randint(4, 200, (12,), dtype=torch.int64),
        torch.tensor([0] * 8 + [1] * 4, dtype=torch.int64),
    )
    long = (
        torch.randint(4, 200, (180,), dtype=torch.int64),
        torch.tensor([0] * 176 + [2] * 4, dtype=torch.int64),
    )

    def _grads_for(mode_kind: str) -> list[torch.Tensor]:
        torch.manual_seed(99)
        m = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, causal=True)
        m.zero_grad(set_to_none=True)
        if mode_kind == "unpacked_avg":
            b1 = collate_padded_pairs([short])
            b2 = collate_padded_pairs([long])
            l1 = packed_sequence_ce_loss(
                m(b1.input_ids, b1.position_ids, seq_id=b1.seq_id, mode="dense"),
                b1.labels,
                loss_reduction="sequence_mean",
                seq_id=b1.seq_id,
            )
            l2 = packed_sequence_ce_loss(
                m(b2.input_ids, b2.position_ids, seq_id=b2.seq_id, mode="dense"),
                b2.labels,
                loss_reduction="sequence_mean",
                seq_id=b2.seq_id,
            )
            loss = 0.5 * (l1 + l2)
        elif mode_kind == "packed_ab":
            pk = pack_sequence_pairs([short, long], pack_len=256, rows_per_batch=1, attn_mode="dense")[0]
            logits = m(pk.input_ids, pk.position_ids, seq_id=pk.seq_id, mode="dense")
            loss = packed_sequence_ce_loss(logits, pk.labels, loss_reduction="sequence_mean", seq_id=pk.seq_id)
        elif mode_kind == "packed_ba":
            pk = pack_sequence_pairs([long, short], pack_len=256, rows_per_batch=1, attn_mode="dense")[0]
            logits = m(pk.input_ids, pk.position_ids, seq_id=pk.seq_id, mode="dense")
            loss = compute_packed_loss(logits, pk.labels, loss_reduction="sequence_mean", seq_id=pk.seq_id)
        else:
            pv = pack_sequence_pairs([short, long], pack_len=256, rows_per_batch=1, attn_mode="packed_varlen")[0]
            logits = m(pv.input_ids, pv.position_ids, cu_seqlens=pv.cu_seqlens, mode="packed_varlen")
            loss = packed_sequence_ce_loss(
                logits, pv.labels, loss_reduction="sequence_mean", cu_seqlens=pv.cu_seqlens
            )
        loss.backward()
        return [p.grad.detach().clone() for p in m.parameters() if p.grad is not None]

    g_unpacked = _grads_for("unpacked_avg")
    g_ab = _grads_for("packed_ab")
    g_ba = _grads_for("packed_ba")
    g_pv = _grads_for("packed_varlen")

    for gu, gab, gba, gpv in zip(g_unpacked, g_ab, g_ba, g_pv, strict=True):
        assert torch.allclose(gab, gu, atol=1e-5)
        assert torch.allclose(gba, gu, atol=1e-5)
        assert torch.allclose(gpv, gu, atol=1e-5)


def test_packed_sequence_ce_loss_curriculum_weighted_and_fixed_horizon() -> None:
    from community_extensions import PackedCurriculumBin, PackedSequenceSlice, materialize_bf16_packed_batch

    pbin = PackedCurriculumBin(
        bin_cap=32,
        slices=(
            PackedSequenceSlice(seq_id=0, offset=0, length=4, curriculum_weight=0.25),
            PackedSequenceSlice(seq_id=1, offset=4, length=16, curriculum_weight=0.0),
            PackedSequenceSlice(seq_id=2, offset=20, length=8, curriculum_weight=0.25),
        ),
    )
    mat = materialize_bf16_packed_batch([pbin], bin_cap=32)
    logits = torch.randn(1, 32, 8, requires_grad=True)
    labels = torch.full((1, 32), -100, dtype=torch.int64)
    labels[0, :28] = torch.randint(0, 8, (28,), dtype=torch.int64)

    loss = packed_sequence_ce_loss(
        logits,
        labels,
        loss_reduction="curriculum_weighted",
        seq_id=mat["seq_ids"],
        token_weights=mat["curriculum_weights_bf16"],
    )
    loss.backward()
    assert logits.grad is not None
    # Zero-variance middle sequence [4:20] must receive exact zero gradient!
    assert float(logits.grad[0, 4:20].abs().max().item()) == 0.0
    assert float(logits.grad[0, :4].abs().max().item()) > 0.0
    assert float(logits.grad[0, 20:28].abs().max().item()) > 0.0

    # Fixed-horizon reduction scales inversely with fixed_denom
    l_h100 = packed_sequence_ce_loss(logits.detach(), labels, loss_reduction="fixed_horizon", fixed_denom=100.0)
    l_h200 = packed_sequence_ce_loss(logits.detach(), labels, loss_reduction="fixed_horizon", fixed_denom=200.0)
    assert torch.allclose(l_h100, 2.0 * l_h200, atol=1e-6)


def test_protocol_transformer_rejects_oversized_position_ids_and_chunks_cleanly() -> None:
    model = ProtocolTransformer(max_seq_len=512, d_model=32, n_heads=2, n_layers=1, d_ff=64, causal=False).eval()
    tok_700 = torch.randint(4, 200, (700,), dtype=torch.int64)
    lbl_700 = torch.zeros(700, dtype=torch.int64)
    over_batch = collate_padded_pairs([(tok_700, lbl_700)])

    with pytest.raises(ValueError, match="max_seq_len"):
        model(over_batch.input_ids, over_batch.position_ids, seq_id=over_batch.seq_id)

    neg_pos = torch.tensor([[-1, 0, 1, 2]], dtype=torch.int64)
    with pytest.raises(ValueError, match="out of range"):
        model(torch.ones((1, 4), dtype=torch.int64), neg_pos)

    with torch.no_grad():
        chunked_logits = model(
            over_batch.input_ids,
            over_batch.position_ids,
            seq_id=over_batch.seq_id,
            chunk_oversized=True,
        )
    assert chunked_logits.shape == (1, 700, model.num_classes)

    chunks = chunk_sequence_pairs([(tok_700, lbl_700)], chunk_len=512)
    assert [int(t.numel()) for t, _ in chunks] == [512, 188]

    pk_chunked = pack_sequence_pairs([(tok_700, lbl_700)], pack_len=512, rows_per_batch=2, overflow_policy="chunk")
    assert sum(b.real_tokens for b in pk_chunked) == 700

    with pytest.raises(ValueError, match="length >= 1"):
        pack_sequence_pairs([(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))], pack_len=512)


def test_detect_gpu_peak_flops_profiler_fields_and_varlen_bf16_ablation() -> None:
    a100_fl, a100_desc = detect_gpu_peak_flops("NVIDIA A100-SXM4-80GB", precision="bf16")
    assert a100_fl == pytest.approx(312.0e12)
    assert "312.00 TFLOPS" in a100_desc

    h100_sxm, _ = detect_gpu_peak_flops("NVIDIA H100 80GB HBM3", precision="bf16")
    assert h100_sxm == pytest.approx(989.5e12)

    h100_pcie, _ = detect_gpu_peak_flops("NVIDIA H100 PCIe", precision="bf16")
    assert h100_pcie == pytest.approx(756.5e12)

    l40s_fl, _ = detect_gpu_peak_flops("NVIDIA L40S", precision="bf16")
    assert l40s_fl == pytest.approx(362.05e12)

    unk_fl, unk_desc = detect_gpu_peak_flops("Mystery Custom Accelerator")
    assert unk_fl is None
    assert "unknown GPU peak" in unk_desc

    ovr_fl, ovr_desc = detect_gpu_peak_flops(None, override_tflops=250.0)
    assert ovr_fl == pytest.approx(250.0e12)
    assert "manual override" in ovr_desc

    # Cached BlockMask reuse in flex_local_global_attention
    sid = _seq_id_row([64, 96], 256)
    bm = build_flex_block_mask(sid, block_size=64, num_global_tokens=4, causal=False)
    q = torch.randn(1, 2, 256, 16)
    out_cached = flex_local_global_attention(q, q, q, seq_id=sid, block_mask=bm, compile_flex=False)
    assert out_cached.shape == (1, 2, 256, 16)

    # Profiler summary includes self_cuda_ms alongside self_cpu_ms
    tiny_model = ProtocolTransformer(d_model=32, n_heads=2, n_layers=1, d_ff=64)
    tiny_batch = collate_padded_pairs([(torch.ones(16, dtype=torch.int64), torch.zeros(16, dtype=torch.int64))])
    prof_ops = run_torch_profiler_summary(tiny_model, tiny_batch)
    assert len(prof_ops) > 0
    assert "self_cpu_ms" in prof_ops[0]
    assert "self_cuda_ms" in prof_ops[0]

    # Extended ablation with include_varlen_bf16=True and calibrated peak_flops
    rows_ext, _, _ = run_fill_gpu_ablation(
        num_sequences=16,
        batch_size=8,
        pack_len=256,
        pack_rows_per_batch=1,
        d_model=32,
        n_layers=1,
        n_heads=2,
        slow_work_iters=5,
        peak_flops=10.0e12,
        seed=0,
        include_varlen_bf16=True,
    )
    by_cfg = {r.config: r for r in rows_ext}
    assert "e2_packed_varlen_bf16" in by_cfg
    assert "f2_varlen_bf16_compile" in by_cfg
    assert by_cfg["e2_packed_varlen_bf16"].status == "ok"
    assert by_cfg["e2_packed_varlen_bf16"].useful_token_fraction == 1.0
    assert by_cfg["a_padded_baseline"].mfu.endswith("%")
    assert rows_ext[-1].config == "i_cuda_graphs"


def test_flash_attn_varlen_func_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from labfill.fill_gpu import TransformerBlock

    blk = TransformerBlock(d_model=32, n_heads=2, d_ff=64, causal=True)
    called_args: dict[str, object] = {}

    class _FakeCudaTensor:
        pass

    def _mock_flash_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        causal: bool = False,
    ) -> torch.Tensor:
        called_args["q_shape"] = tuple(q.shape)
        called_args["cu_dtype"] = cu_q.dtype
        called_args["max_q"] = max_q
        called_args["max_k"] = max_k
        called_args["causal"] = causal
        return v.clone()

    monkeypatch.setattr(fill_gpu_mod, "_FLASH_ATTN_VARLEN_FUNC", _mock_flash_varlen)

    # Verify fallback when on CPU (even with bf16 dtype, flash_attn_varlen_func only runs on CUDA)
    x_cpu = torch.randn(10, 32, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4, 10], dtype=torch.int64)
    out_cpu = blk._run_varlen_qkv(x_cpu, x_cpu, x_cpu, cu, max_seqlen=6)
    assert out_cpu.shape == (10, 32)
    assert called_args == {}


def test_obfd_segment_tree_matches_naive_bfd_and_scales_to_10k_sequences() -> None:
    """Verify Ding et al. (2024) O(N log N) OBFD matches O(N^2) BFD bit-for-bit and scales fast."""
    import time

    def _naive_bfd(lengths: list[int], bin_cap: int) -> list[list[tuple[int, int, int]]]:
        order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
        used: list[int] = []
        bins: list[list[tuple[int, int, int]]] = []
        for idx in order:
            ln = lengths[idx]
            best_b = -1
            best_rem = bin_cap + 1
            for b_i, u in enumerate(used):
                rem = bin_cap - u
                if ln <= rem and (rem - ln) < best_rem:
                    best_rem = rem - ln
                    best_b = b_i
            if best_b < 0:
                used.append(ln)
                bins.append([(idx, 0, ln)])
            else:
                off = used[best_b]
                bins[best_b].append((idx, off, ln))
                used[best_b] = off + ln
        return bins

    for seed in (0, 17, 99):
        items = generate_protocol_sequences(num_sequences=180, min_len=8, max_len=512, seed=seed)
        lens = [len(it.tokens) for it in items]
        obfd = pack_lengths_obfd(lens, bin_cap=512)
        naive = _naive_bfd(lens, bin_cap=512)
        assert obfd == naive

        pairs = [
            (torch.tensor(it.tokens, dtype=torch.int64), torch.tensor(it.labels, dtype=torch.int64))
            for it in items
        ]
        ff_batches = pack_sequence_pairs(pairs, pack_len=512, rows_per_batch=1, attn_mode="dense")
        obfd_batches = pack_sequences_obfd(pairs, pack_len=512, rows_per_batch=1, attn_mode="dense")
        assert len(obfd_batches) <= len(ff_batches)
        assert sum(b.real_tokens for b in obfd_batches) == sum(lens)

    # Scale check: 10,000 sequences packed into bin_cap=2048 in well under 0.5s
    g = torch.Generator().manual_seed(2026)
    big_lens = torch.randint(16, 1500, (10_000,), generator=g).tolist()
    t0 = time.perf_counter()
    big_bins = pack_lengths_obfd(big_lens, bin_cap=2048)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0
    assert sum(ln for b in big_bins for _, _, ln in b) == sum(big_lens)


