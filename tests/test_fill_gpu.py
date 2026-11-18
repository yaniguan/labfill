from __future__ import annotations

import torch

from labfill.fill_gpu import (
    ProtocolTransformer,
    collate_padded_pairs,
    collate_varlen_pairs,
    count_batch_training_flops,
    pack_sequence_pairs,
    run_fill_gpu_ablation,
)


def test_sequence_packing_and_varlen_no_cross_contamination() -> None:
    torch.manual_seed(0)
    model = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128, causal=False)
    model.eval()

    seq_a = (torch.tensor([1, 17, 33, 49, 65, 2], dtype=torch.int64), torch.zeros(6, dtype=torch.int64))
    seq_b1 = (torch.tensor([1, 18, 34, 50, 66, 70, 80, 2], dtype=torch.int64), torch.zeros(8, dtype=torch.int64))
    seq_b2 = (torch.tensor([1, 200, 201, 202, 203, 204, 205, 2], dtype=torch.int64), torch.zeros(8, dtype=torch.int64))

    # 1. Dense packed mode: mutating seq_b tokens must NOT alter seq_a logits
    pk1 = pack_sequence_pairs([seq_a, seq_b1], pack_len=128, rows_per_batch=1, attn_mode="dense")[0]
    pk2 = pack_sequence_pairs([seq_a, seq_b2], pack_len=128, rows_per_batch=1, attn_mode="dense")[0]
    # Verify position_ids reset to 0 at start of seq_b (index 6)
    assert pk1.position_ids[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert pk1.position_ids[0, 6:14].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

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

        # 3. Jagged varlen mode matches standalone single-sequence forward pass
        vl = collate_varlen_pairs([seq_a, seq_b1])
        out_vl = model(vl.input_ids, vl.position_ids, cu_seqlens=vl.cu_seqlens, mode="varlen")
        single_a = collate_padded_pairs([seq_a])
        out_single = model(single_a.input_ids, single_a.position_ids, seq_id=single_a.seq_id, mode="dense")
        assert torch.allclose(out_vl[:6], out_single[0, :6], atol=1e-5)


def test_analytic_flop_accounting() -> None:
    model = ProtocolTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=256)
    short = (torch.ones(10, dtype=torch.int64), torch.zeros(10, dtype=torch.int64))
    long = (torch.ones(100, dtype=torch.int64), torch.zeros(100, dtype=torch.int64))

    padded = collate_padded_pairs([short, long])
    u_pad, e_pad = count_batch_training_flops(model, padded)
    assert 0.0 < u_pad < e_pad

    vl = collate_varlen_pairs([short, long])
    u_vl, e_vl = count_batch_training_flops(model, vl)
    assert abs(u_vl - e_vl) < 1e-6
    assert abs(u_vl - u_pad) < 1e-6


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
    by_name = {r.config: r for r in rows}
    assert by_name["a_padded_baseline"].status == "ok"
    assert by_name["b_length_bucketed"].useful_token_fraction > by_name["a_padded_baseline"].useful_token_fraction
    assert by_name["d_varlen_nested_sdpa"].useful_token_fraction == 1.0
    assert by_name["e_packed_bf16_autocast"].status == "ok"
    assert by_name["g2_async_dataloader"].status == "ok"
    assert by_name["h_block_sparse_bf16"].status == "ok"
    assert by_name["a_padded_baseline"].mfu == "N/A"
    assert len(prof_ops) > 0
    assert "forward_ok" in flex_info
