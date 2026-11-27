"""Tests for the Auto-Lab 15-repository wire node (`autolab_wire.py`, Layer L3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import autolab_wire  # noqa: E402
from autolab_wire import (  # noqa: E402
    LAYER,
    REPO_ID,
    _pack_with_labfill,
    pack_lengths_pure_python,
    packed_batch_to_posttrain_tensors,
    process_packet,
)


def test_wire_metadata_and_default_conservation() -> None:
    assert REPO_ID == "labfill"
    assert LAYER == "L3"
    assert autolab_wire._HAS_TORCH_LABFILL is True

    raw_packet: dict[str, object] = {}
    out = process_packet(raw_packet)
    assert raw_packet == {}

    assert out["schema_version"] == "autolab-wire/v1"
    assert out["halted"] is False
    assert out["halt_reason"] is None

    pb = out["packed_batch"]
    assert pb["bin_cap"] == 512
    assert pb["bin_count"] == 1
    assert pb["total_real_tokens"] == 492
    assert pb["useful_token_ratio"] == round(492 / 512, 4)
    assert pb["zero_cross_seq_overlap"] is True

    assert len(out["trace"]) == 1
    assert out["trace"][0]["repo"] == "labfill"
    assert out["trace"][0]["layer"] == "L3"
    assert out["trace"][0]["status"] == "PASS"

    assert len(out["bottom_reasoning_traces"]) == 1
    assert out["bottom_reasoning_traces"][0]["repo"] == "labfill"
    assert out["bottom_reasoning_traces"][0]["invariant"] == "checkPackedBin_conserves_tokens"


def test_error_twins_multi_bin_packing_conserves_tokens() -> None:
    lengths = [220, 190, 180, 150, 96]
    pkt = {
        "error_twins": {
            "step_count": 4,
            "corrupted_twin_count": 5,
            "token_lengths": lengths,
        },
        "bin_cap": 512,
    }
    out = process_packet(pkt)
    assert out["halted"] is False

    # Bin 0 packs 220 + 190 + 96 = 506 <= 512; Bin 1 packs 180 + 150 = 330 <= 512
    pb = out["packed_batch"]
    assert pb["bin_cap"] == 512
    assert pb["bin_count"] == 2
    assert pb["total_real_tokens"] == sum(lengths)
    assert pb["useful_token_ratio"] == round(sum(lengths) / (2 * 512), 4)
    assert pb["zero_cross_seq_overlap"] is True


def test_overlapping_slices_and_non_conservation_rejected() -> None:
    # 1. Boolean flag injects overlapping slices into bin 0
    out_flag = process_packet(
        {
            "error_twins": {"token_lengths": [120, 160, 140]},
            "bin_cap": 512,
            "force_overlapping_slices": True,
        }
    )
    assert out_flag["halted"] is True
    assert out_flag["halt_reason"] == "L3_LABFILL_PACK_BLOCK"
    assert out_flag["packed_batch"]["zero_cross_seq_overlap"] is False
    assert out_flag["trace"][-1]["status"] == "BLOCK"
    assert out_flag["bottom_reasoning_traces"][-1]["invariant"] == "checkPackedBin_conserves_tokens"

    # 2. Explicit overlapping slice intervals (sum == 492, within bin_cap=512, but overlapping on [120, 260))
    out_explicit = process_packet(
        {
            "error_twins": {"token_lengths": [260, 232]},
            "bin_cap": 512,
            "force_overlapping_slices": [
                {"seq_id": 1, "offset": 0, "len": 260},
                {"seq_id": 2, "offset": 120, "len": 232},
            ],
        }
    )
    assert out_explicit["halted"] is True
    assert out_explicit["halt_reason"] == "L3_LABFILL_PACK_BLOCK"
    assert out_explicit["packed_batch"]["zero_cross_seq_overlap"] is False

    # 3. Non-overlapping slices whose token sum violates conservation
    out_drop = process_packet(
        {
            "error_twins": {"token_lengths": [120, 140]},
            "bin_cap": 512,
            "force_overlapping_slices": [
                {"seq_id": 1, "offset": 0, "len": 120},
            ],
        }
    )
    assert out_drop["halted"] is True
    assert out_drop["halt_reason"] == "L3_LABFILL_PACK_BLOCK"

    # 4. Single sequence exceeding bin_cap rejected without truncation
    out_overflow = process_packet(
        {
            "error_twins": {"token_lengths": [600, 120]},
            "bin_cap": 512,
        }
    )
    assert out_overflow["halted"] is True
    assert out_overflow["halt_reason"] == "L3_LABFILL_PACK_BLOCK"


def test_halted_packet_passthrough() -> None:
    upstream_halted = {
        "schema_version": "autolab-wire/v1",
        "halted": True,
        "halt_reason": "L4_ATTRIB_QUARANTINE",
        "trace": [
            {
                "repo": "lab-attrib",
                "layer": "L4",
                "status": "BLOCK",
                "summary": "Precursor overlap quarantined.",
            }
        ],
    }
    out = process_packet(upstream_halted)
    assert out["halted"] is True
    assert out["halt_reason"] == "L4_ATTRIB_QUARANTINE"
    assert "packed_batch" not in out
    assert len(out["trace"]) == 1
    assert out["bottom_reasoning_traces"] == []


def test_pure_python_packer_matches_labfill_torch_packer() -> None:
    lengths = [96, 220, 144, 180, 72, 200]
    torch_bins, mask_ok = _pack_with_labfill(lengths, bin_cap=512)
    py_bins = pack_lengths_pure_python(lengths, bin_cap=512)

    assert mask_ok is True
    assert len(torch_bins) == len(py_bins)
    for tb, pb in zip(torch_bins, py_bins, strict=True):
        assert [(s.offset, s.length) for s in tb] == [(s.offset, s.length) for s in pb]


def test_dynamic_spec_loading_without_sys_modules_registration(monkeypatch: object) -> None:
    import importlib.util
    import types

    # Simulate another sibling repo already occupying sys.modules["community_extensions"]
    dummy_ext = types.ModuleType("community_extensions")
    monkeypatch.setitem(sys.modules, "community_extensions", dummy_ext)  # type: ignore[attr-defined]
    monkeypatch.delitem(sys.modules, "_labfill_curriculum_difficulty_packer", raising=False)  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location("unregistered_labfill_wire", REPO_ROOT / "autolab_wire.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = mod.process_packet({"error_twins": {"token_lengths": [100, 150, 120]}})
    assert out["halted"] is False
    assert out["packed_batch"]["total_real_tokens"] == 370
    assert out["packed_batch"]["zero_cross_seq_overlap"] is True


def test_wire_emits_bins_cu_seqlens_loss_mask_spans_and_posttrain_tensors() -> None:
    import pytest
    import torch

    pkt = {
        "error_twins": {
            "token_lengths": [220, 190, 180, 96],
            "prompt_lengths": [140, 110, 120, 40],
            "pass_rates": [0.5, 0.25, 0.0, 0.75],
        },
        "bin_cap": 512,
    }
    out = process_packet(pkt)
    assert out["halted"] is False
    pb = out["packed_batch"]

    # Bin 0 packs seq 0 (220), seq 1 (190), seq 3 (96) -> 506; Bin 1 packs seq 2 (180)
    assert pb["bin_count"] == 2
    assert pb["total_real_tokens"] == 686
    assert pb["total_completion_tokens"] == (80 + 80 + 60 + 56)
    assert pb["cu_seqlens"][-1] == pb["total_real_tokens"]
    assert pb["cu_seqlens"] == [0, 220, 410, 506, 686]
    assert pb["bin_cu_seqlens"] == [[0, 220, 410, 506], [0, 180]]
    assert pb["max_seqlen"] == 220
    assert len(pb["loss_mask_spans"]) == 4

    # Materialize 2D binned tensors and verify prompt-prefix masking + completion sum
    pos_2d, cu_i32, max_s, comp_2d, weights_2d = packed_batch_to_posttrain_tensors(pb)
    assert pos_2d.shape == (2, 512)
    assert cu_i32.dtype == torch.int32
    assert int(cu_i32[-1].item()) == pb["total_real_tokens"]
    assert max_s == 220
    assert int(comp_2d.sum().item()) == pb["total_completion_tokens"]
    assert weights_2d.dtype == torch.bfloat16

    for b_idx, b_slices in enumerate(pb["bins"]):
        for sl in b_slices:
            off = sl["offset"]
            p_len = sl["prompt_len"]
            c_off = sl["completion_offset"]
            assert c_off == off + p_len
            end = off + sl["length"]
            assert bool(comp_2d[b_idx, off:c_off].any().item()) is False
            assert bool(comp_2d[b_idx, c_off:end].all().item()) is True
            assert int(pos_2d[b_idx, off].item()) == 0

    # Materialize 1D flat tensors aligned directly with cu_seqlens
    flat_toks = torch.ones((pb["total_real_tokens"],), dtype=torch.int64)
    pos_1d, _, _, comp_1d, weights_1d = packed_batch_to_posttrain_tensors(pb, token_ids=flat_toks)
    assert pos_1d.shape == (686,)
    assert int(comp_1d.sum().item()) == pb["total_completion_tokens"]
    assert weights_1d.dtype == torch.bfloat16

    # Optional drop_zero_variance=True drops seq 2 (pass_rate == 0.0 -> weight == 0.0)
    out_drop_zv = process_packet({**pkt, "drop_zero_variance": True})
    assert out_drop_zv["halted"] is False
    pb_zv = out_drop_zv["packed_batch"]
    assert pb_zv["zero_variance_dropped"] == 1
    assert pb_zv["total_real_tokens"] == 220 + 190 + 96
    assert pb_zv["bin_count"] == 1

    # Rollout rewards directly compute Bernoulli variance p * (1 - p)
    out_rw = process_packet(
        {
            "error_twins": {
                "token_lengths": [100, 120],
                "rollout_rewards": [[1, 0, 1, 0], [1, 1, 1, 0]],
            },
            "bin_cap": 512,
        }
    )
    assert out_rw["halted"] is False
    assert out_rw["packed_batch"]["bins"][0][0]["curriculum_weight"] == pytest.approx(0.25)
    assert out_rw["packed_batch"]["bins"][0][1]["curriculum_weight"] == pytest.approx(0.1875)


def test_wire_overflow_policies_zero_ratio_on_halt_and_large_bin_cap() -> None:
    import pytest

    # 1. Explicit bin_cap=512 overflow halts with useful_token_ratio == 0.0 and bin_count == 0
    halted_over = process_packet({"error_twins": {"token_lengths": [600, 120]}, "bin_cap": 512})
    assert halted_over["halted"] is True
    assert halted_over["packed_batch"]["bin_count"] == 0
    assert halted_over["packed_batch"]["useful_token_ratio"] == 0.0
    assert halted_over["packed_batch"]["bins"] == []
    assert halted_over["packed_batch"]["cu_seqlens"] == [0]
    with pytest.raises(ValueError, match="halted"):
        packed_batch_to_posttrain_tensors(halted_over["packed_batch"])

    # 2. Implicit bin_cap auto-aligns 104-step plate protocol (522 tokens) to LabFillLean 2048 cap
    plate_pkt = process_packet({"error_twins": {"token_lengths": [522, 120, 144]}})
    assert plate_pkt["halted"] is False
    assert plate_pkt["packed_batch"]["bin_cap"] == 2048
    assert plate_pkt["packed_batch"]["total_real_tokens"] == 786

    # 3. Quarantine policy isolates oversized sequence without aborting valid sequences
    quar_pkt = process_packet(
        {
            "error_twins": {"token_lengths": [700, 180, 140]},
            "bin_cap": 512,
            "overflow_policy": "quarantine_oversized",
        }
    )
    assert quar_pkt["halted"] is False
    assert quar_pkt["packed_batch"]["quarantined_seq_indices"] == [0]
    assert quar_pkt["packed_batch"]["total_real_tokens"] == 320

    # 4. Chunk policy splits oversized sequence cleanly across bins
    chunk_pkt = process_packet(
        {
            "error_twins": {"token_lengths": [700, 120], "prompt_lengths": [600, 40]},
            "bin_cap": 512,
            "overflow_policy": "chunk",
        }
    )
    assert chunk_pkt["halted"] is False
    assert chunk_pkt["packed_batch"]["total_real_tokens"] == 820
    assert chunk_pkt["packed_batch"]["total_completion_tokens"] == 100 + 80

    # 5. Out-of-bounds prompt_lengths (> token_len or < 0) halts cleanly
    bad_prompt = process_packet(
        {
            "error_twins": {"token_lengths": [120, 140], "prompt_lengths": [150, 20]},
            "bin_cap": 512,
        }
    )
    assert bad_prompt["halted"] is True
    assert bad_prompt["halt_reason"] == "L3_LABFILL_PACK_BLOCK"

    # 6. Large bin_cap = 16384 runs in O(bin_cap) without dense O(bin_cap^2) boolean mask blowup
    large_cap = process_packet(
        {
            "error_twins": {"token_lengths": [100, 120, 90, 110]},
            "bin_cap": 16384,
        }
    )
    assert large_cap["halted"] is False
    assert large_cap["packed_batch"]["total_real_tokens"] == 420
    assert large_cap["packed_batch"]["cu_seqlens"] == [0, 100, 220, 310, 420]


