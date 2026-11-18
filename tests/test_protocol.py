from __future__ import annotations

import torch

from labfill.protocol import (
    OP_PICK_TIP,
    OP_TRANSFER,
    PIP_NONE,
    PIP_P2,
    PIP_P20,
    PIP_P200,
    PIP_P1000,
    ProtocolBatch,
    make_pcr_protocol,
    make_random_valid_protocols,
    pick_pipette_for_vol,
    ul,
)


def test_units_and_pipette_selection() -> None:
    assert ul(0.2) == 2
    assert ul(2.5) == 25
    assert ul(200.0) == 2000
    assert pick_pipette_for_vol(ul(0.5)) == PIP_P2
    assert pick_pipette_for_vol(ul(5.0)) == PIP_P20
    assert pick_pipette_for_vol(ul(50.0)) == PIP_P200
    assert pick_pipette_for_vol(ul(500.0)) == PIP_P1000
    assert pick_pipette_for_vol(1) == PIP_NONE
    assert pick_pipette_for_vol(20000) == PIP_NONE


def test_batch_helpers_and_state_tensor() -> None:
    st, pr = make_random_valid_protocols(batch_size=4, max_steps=16, num_containers=6, seed=0)
    assert st.batch_size == 4
    assert st.num_containers == 6
    assert pr.batch_size == 4
    assert pr.max_steps == 16

    packed = st.as_state_tensor()
    assert packed.shape == (4, 6, 4)

    rep_st = st.repeat_batch(3)
    rep_pr = pr.repeat_batch(3)
    assert rep_st.batch_size == 12
    assert rep_pr.batch_size == 12

    idx = torch.tensor([0, 2], dtype=torch.int64, device=st.device)
    assert st.index_select(idx).batch_size == 2
    assert pr.index_select(idx).batch_size == 2


def test_from_steps_list_and_pcr() -> None:
    batch = ProtocolBatch.from_steps_list(
        [
            [{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "src": 0, "dst": 1, "vol": 50, "pipette": PIP_P20}],
            [{"op": OP_PICK_TIP}],
        ]
    )
    assert batch.batch_size == 2
    assert batch.max_steps == 2
    assert bool(batch.pad_mask[1, 1].item()) is True
    assert bool(batch.pad_mask[0, 1].item()) is False

    st_pcr, pr_pcr = make_pcr_protocol()
    assert st_pcr.batch_size == 1
    assert pr_pcr.max_steps > 20
