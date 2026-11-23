from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from labfill.fill_gpu import (
    CudaGraphStepRunner,
    PackCollate,
    ProtocolTransformer,
    SlowTokenizerDataset,
    _sequence_start_offsets,
    build_batches_from_items,
    build_block_diag_mask,
    collate_padded_pairs,
    collate_varlen_pairs,
    count_batch_training_flops,
    flex_local_global_attention,
    generate_protocol_sequences,
    pack_sequence_pairs,
    probe_flex_attention,
    run_fill_gpu_ablation,
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
