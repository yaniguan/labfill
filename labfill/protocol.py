"""Core tensor encoding for laboratory protocols and container state.

Units match LabLean:
- Volume: integer 0.1 uL units (e.g. 2.5 uL -> 25).
- Temperature: whole degrees Celsius.
- Duration: whole seconds.
- Centrifuge speed: whole RPM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

# Op codes
OP_NOOP = 0
OP_PICK_TIP = 1
OP_DROP_TIP = 2
OP_TRANSFER = 3
OP_MIX = 4
OP_INCUBATE = 5
OP_THAW = 6
OP_SEAL = 7
OP_UNSEAL = 8
OP_CENTRIFUGE = 9

OP_NAMES: dict[int, str] = {
    OP_NOOP: "noop",
    OP_PICK_TIP: "pick_tip",
    OP_DROP_TIP: "drop_tip",
    OP_TRANSFER: "transfer",
    OP_MIX: "mix",
    OP_INCUBATE: "incubate",
    OP_THAW: "thaw",
    OP_SEAL: "seal",
    OP_UNSEAL: "unseal",
    OP_CENTRIFUGE: "centrifuge",
}

# Pipette ids and ranges in 0.1 uL integer units (matches LabLean)
PIP_NONE = 0
PIP_P2 = 1
PIP_P20 = 2
PIP_P200 = 3
PIP_P1000 = 4

PIPETTE_NAMES: dict[int, str] = {
    PIP_NONE: "NONE",
    PIP_P2: "P2",
    PIP_P20: "P20",
    PIP_P200: "P200",
    PIP_P1000: "P1000",
}

PIPETTE_RANGES_TENTHS_UL: dict[int, tuple[int, int]] = {
    PIP_P2: (2, 20),          # 0.2 .. 2.0 uL
    PIP_P20: (20, 200),       # 2.0 .. 20.0 uL
    PIP_P200: (200, 2000),    # 20.0 .. 200.0 uL
    PIP_P1000: (1000, 10000), # 100.0 .. 1000.0 uL
}

# Error codes (0 = valid)
ERR_OK = 0
ERR_BAD_OP = 1
ERR_BAD_CONTAINER = 2
ERR_NO_TIP = 3
ERR_TIP_ALREADY_ON = 4
ERR_BAD_PIPETTE = 5
ERR_SAME_CONTAINER = 6
ERR_SEALED = 7
ERR_NOT_SEALED = 8
ERR_FROZEN = 9
ERR_CONTAMINATED = 10
ERR_NOT_ENOUGH = 11
ERR_OVERFLOW = 12
ERR_BAD_TEMP = 13
ERR_BAD_RPM = 14
ERR_UNBALANCED = 15
ERR_BAD_DURATION = 16

ERR_NAMES: dict[int, str] = {
    ERR_OK: "ok",
    ERR_BAD_OP: "bad_op",
    ERR_BAD_CONTAINER: "bad_container",
    ERR_NO_TIP: "no_tip",
    ERR_TIP_ALREADY_ON: "tip_already_on",
    ERR_BAD_PIPETTE: "bad_pipette",
    ERR_SAME_CONTAINER: "same_container",
    ERR_SEALED: "sealed",
    ERR_NOT_SEALED: "not_sealed",
    ERR_FROZEN: "frozen",
    ERR_CONTAMINATED: "contaminated",
    ERR_NOT_ENOUGH: "not_enough",
    ERR_OVERFLOW: "overflow",
    ERR_BAD_TEMP: "bad_temp",
    ERR_BAD_RPM: "bad_rpm",
    ERR_UNBALANCED: "unbalanced",
    ERR_BAD_DURATION: "bad_duration",
}

ROTOR_TOL_TENTHS_UL = 50  # 5.0 uL balance tolerance, matching LabLean
MAX_RPM = 15000
MIN_INCUBATE_C = 4
MAX_INCUBATE_C = 98
HIGH_TEMP_SEAL_C = 80     # >= 80 C requires sealed container to prevent evaporation
AMT_SCALE = 10_000        # 1e-5 uL solute tracking unit, matching LabLean amtScale
CONC_TOL_PERMILLE = 1     # 0.1% concentration tolerance, matching LabLean concTolPermille


def default_device(device: str | torch.device | None = None) -> torch.device:
    """Return requested device, or 'cuda' when available else 'cpu'."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ul(microliters: float) -> int:
    """Convert microliters to 0.1 uL integer units."""
    return int(round(microliters * 10.0))


def pick_pipette_for_vol(vol_tenths: int) -> int:
    """Pick the smallest valid LabLean pipette covering vol_tenths (0.1 uL)."""
    for pip_id in (PIP_P2, PIP_P20, PIP_P200, PIP_P1000):
        lo, hi = PIPETTE_RANGES_TENTHS_UL[pip_id]
        if lo <= vol_tenths <= hi:
            return pip_id
    return PIP_NONE


@dataclass
class LabStateBatch:
    """Batched state over B protocols and C containers.

    Tensors:
    - volume: (B, C) int64 in 0.1 uL
    - capacity: (B, C) int64 in 0.1 uL
    - sealed: (B, C) bool
    - frozen: (B, C) bool
    - reagent_mask: (B, C) int64 bitmask of up to 62 reagents per container
    - has_tip: (B,) bool
    - tip_mask: (B,) int64 bitmask of reagents touched by current tip
    """

    volume: Tensor
    capacity: Tensor
    sealed: Tensor
    frozen: Tensor
    reagent_mask: Tensor | None = None
    has_tip: Tensor | None = None
    tip_mask: Tensor | None = None
    diluent_mask: int = 0

    def __post_init__(self) -> None:
        if self.volume.ndim != 2:
            raise ValueError(f"volume must be a 2D (B, C) tensor, got shape {tuple(self.volume.shape)}")
        if self.volume.dtype != torch.int64:
            raise ValueError(f"volume must have dtype int64, got {self.volume.dtype}")
        b, c = self.volume.shape
        dev = self.volume.device

        if self.reagent_mask is None:
            self.reagent_mask = torch.zeros((b, c), dtype=torch.int64, device=dev)
        if self.has_tip is None:
            self.has_tip = torch.zeros((b,), dtype=torch.bool, device=dev)
        if self.tip_mask is None:
            self.tip_mask = torch.zeros((b,), dtype=torch.int64, device=dev)

        for name, ten, exp_shape, exp_dtype in (
            ("capacity", self.capacity, (b, c), torch.int64),
            ("sealed", self.sealed, (b, c), torch.bool),
            ("frozen", self.frozen, (b, c), torch.bool),
            ("reagent_mask", self.reagent_mask, (b, c), torch.int64),
            ("has_tip", self.has_tip, (b,), torch.bool),
            ("tip_mask", self.tip_mask, (b,), torch.int64),
        ):
            if tuple(ten.shape) != exp_shape:
                raise ValueError(f"{name} shape {tuple(ten.shape)} != expected {exp_shape}")
            if ten.dtype != exp_dtype:
                raise ValueError(f"{name} dtype {ten.dtype} != expected {exp_dtype}")
            if ten.device != dev:
                raise ValueError(f"{name} device {ten.device} != volume device {dev}")

        if not isinstance(self.diluent_mask, int) or isinstance(self.diluent_mask, bool) or self.diluent_mask < 0:
            raise ValueError(f"diluent_mask must be a non-negative int bitmask, got {self.diluent_mask!r}")

    @property
    def batch_size(self) -> int:
        return int(self.volume.shape[0])

    @property
    def num_containers(self) -> int:
        return int(self.volume.shape[1])

    @property
    def device(self) -> torch.device:
        return self.volume.device

    def clone(self) -> "LabStateBatch":
        return LabStateBatch(
            volume=self.volume.clone(),
            capacity=self.capacity.clone(),
            sealed=self.sealed.clone(),
            frozen=self.frozen.clone(),
            reagent_mask=self.reagent_mask.clone() if self.reagent_mask is not None else None,
            has_tip=self.has_tip.clone() if self.has_tip is not None else None,
            tip_mask=self.tip_mask.clone() if self.tip_mask is not None else None,
            diluent_mask=self.diluent_mask,
        )

    def to(self, device: str | torch.device) -> "LabStateBatch":
        dev = torch.device(device)
        return LabStateBatch(
            volume=self.volume.to(dev),
            capacity=self.capacity.to(dev),
            sealed=self.sealed.to(dev),
            frozen=self.frozen.to(dev),
            reagent_mask=self.reagent_mask.to(dev) if self.reagent_mask is not None else None,
            has_tip=self.has_tip.to(dev) if self.has_tip is not None else None,
            tip_mask=self.tip_mask.to(dev) if self.tip_mask is not None else None,
            diluent_mask=self.diluent_mask,
        )

    def repeat_batch(self, n: int) -> "LabStateBatch":
        """Repeat a B-sized state batch n times along batch dim -> (B * n, C)."""
        return LabStateBatch(
            volume=self.volume.repeat_interleave(n, dim=0),
            capacity=self.capacity.repeat_interleave(n, dim=0),
            sealed=self.sealed.repeat_interleave(n, dim=0),
            frozen=self.frozen.repeat_interleave(n, dim=0),
            reagent_mask=self.reagent_mask.repeat_interleave(n, dim=0),
            has_tip=self.has_tip.repeat_interleave(n, dim=0),
            tip_mask=self.tip_mask.repeat_interleave(n, dim=0),
            diluent_mask=self.diluent_mask,
        )

    def index_select(self, idx: Tensor) -> "LabStateBatch":
        return LabStateBatch(
            volume=self.volume[idx],
            capacity=self.capacity[idx],
            sealed=self.sealed[idx],
            frozen=self.frozen[idx],
            reagent_mask=self.reagent_mask[idx],
            has_tip=self.has_tip[idx],
            tip_mask=self.tip_mask[idx],
            diluent_mask=self.diluent_mask,
        )

    def as_state_tensor(self) -> Tensor:
        """Pack [volume, capacity, sealed, frozen] into a (B, C, 4) int64 tensor."""
        return torch.stack(
            [
                self.volume,
                self.capacity,
                self.sealed.to(torch.int64),
                self.frozen.to(torch.int64),
            ],
            dim=-1,
        )


@dataclass
class ProtocolBatch:
    """Batched tensor encoding of B protocols of length up to T steps.

    All step fields have shape (B, T) int64; pad_mask has shape (B, T) bool
    (True = padded slot, ignored by checkers and simulators).
    """

    op: Tensor
    src: Tensor
    dst: Tensor
    vol: Tensor
    temp: Tensor
    duration: Tensor
    pipette: Tensor
    rpm: Tensor | None = None
    pad_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.op.ndim != 2:
            raise ValueError(f"op must be a 2D (B, T) tensor, got shape {tuple(self.op.shape)}")
        if self.op.dtype != torch.int64:
            raise ValueError(f"op must have dtype int64, got {self.op.dtype}")
        b, t = self.op.shape
        dev = self.op.device
        if self.rpm is None:
            self.rpm = torch.zeros((b, t), dtype=torch.int64, device=dev)
        if self.pad_mask is None:
            self.pad_mask = torch.zeros((b, t), dtype=torch.bool, device=dev)

        for name, ten, exp_dtype in (
            ("src", self.src, torch.int64),
            ("dst", self.dst, torch.int64),
            ("vol", self.vol, torch.int64),
            ("temp", self.temp, torch.int64),
            ("duration", self.duration, torch.int64),
            ("pipette", self.pipette, torch.int64),
            ("rpm", self.rpm, torch.int64),
            ("pad_mask", self.pad_mask, torch.bool),
        ):
            if tuple(ten.shape) != (b, t):
                raise ValueError(f"{name} shape {tuple(ten.shape)} != expected {(b, t)}")
            if ten.dtype != exp_dtype:
                raise ValueError(f"{name} dtype {ten.dtype} != expected {exp_dtype}")
            if ten.device != dev:
                raise ValueError(f"{name} device {ten.device} != op device {dev}")

    @property
    def batch_size(self) -> int:
        return int(self.op.shape[0])

    @property
    def max_steps(self) -> int:
        return int(self.op.shape[1])

    @property
    def device(self) -> torch.device:
        return self.op.device

    def clone(self) -> "ProtocolBatch":
        return ProtocolBatch(
            op=self.op.clone(),
            src=self.src.clone(),
            dst=self.dst.clone(),
            vol=self.vol.clone(),
            temp=self.temp.clone(),
            duration=self.duration.clone(),
            pipette=self.pipette.clone(),
            rpm=self.rpm.clone() if self.rpm is not None else None,
            pad_mask=self.pad_mask.clone() if self.pad_mask is not None else None,
        )

    def to(self, device: str | torch.device) -> "ProtocolBatch":
        dev = torch.device(device)
        return ProtocolBatch(
            op=self.op.to(dev),
            src=self.src.to(dev),
            dst=self.dst.to(dev),
            vol=self.vol.to(dev),
            temp=self.temp.to(dev),
            duration=self.duration.to(dev),
            pipette=self.pipette.to(dev),
            rpm=self.rpm.to(dev) if self.rpm is not None else None,
            pad_mask=self.pad_mask.to(dev) if self.pad_mask is not None else None,
        )

    def repeat_batch(self, n: int) -> "ProtocolBatch":
        return ProtocolBatch(
            op=self.op.repeat_interleave(n, dim=0),
            src=self.src.repeat_interleave(n, dim=0),
            dst=self.dst.repeat_interleave(n, dim=0),
            vol=self.vol.repeat_interleave(n, dim=0),
            temp=self.temp.repeat_interleave(n, dim=0),
            duration=self.duration.repeat_interleave(n, dim=0),
            pipette=self.pipette.repeat_interleave(n, dim=0),
            rpm=self.rpm.repeat_interleave(n, dim=0),
            pad_mask=self.pad_mask.repeat_interleave(n, dim=0),
        )

    def index_select(self, idx: Tensor) -> "ProtocolBatch":
        return ProtocolBatch(
            op=self.op[idx],
            src=self.src[idx],
            dst=self.dst[idx],
            vol=self.vol[idx],
            temp=self.temp[idx],
            duration=self.duration[idx],
            pipette=self.pipette[idx],
            rpm=self.rpm[idx],
            pad_mask=self.pad_mask[idx],
        )

    @classmethod
    def from_steps_list(
        cls,
        protocols: Sequence[Sequence[dict[str, Any]]],
        max_steps: int | None = None,
        device: str | torch.device | None = None,
        truncate: bool = False,
    ) -> "ProtocolBatch":
        """Build a ProtocolBatch from Python lists of step dicts.

        Raises ValueError if any protocol exceeds `max_steps` unless `truncate=True`.
        """
        dev = default_device(device)
        b = len(protocols)
        if max_steps is not None:
            if max_steps < 1:
                raise ValueError(f"max_steps must be >= 1, got {max_steps}")
            if not truncate:
                for i, steps in enumerate(protocols):
                    if len(steps) > max_steps:
                        raise ValueError(
                            f"Protocol {i} has {len(steps)} steps, exceeding max_steps={max_steps}; "
                            "pass truncate=True to allow truncation"
                        )
            t = max_steps
        else:
            t = max(max((len(p) for p in protocols), default=1), 1)

        op = torch.zeros((b, t), dtype=torch.int64)
        src = torch.zeros((b, t), dtype=torch.int64)
        dst = torch.zeros((b, t), dtype=torch.int64)
        vol = torch.zeros((b, t), dtype=torch.int64)
        temp = torch.zeros((b, t), dtype=torch.int64)
        duration = torch.zeros((b, t), dtype=torch.int64)
        pipette = torch.zeros((b, t), dtype=torch.int64)
        rpm = torch.zeros((b, t), dtype=torch.int64)
        pad_mask = torch.ones((b, t), dtype=torch.bool)

        for i, steps in enumerate(protocols):
            for j, st in enumerate(steps[:t]):
                pad_mask[i, j] = False
                op[i, j] = int(st.get("op", OP_NOOP))
                src[i, j] = int(st.get("src", 0))
                dst[i, j] = int(st.get("dst", 0))
                vol[i, j] = int(st.get("vol", 0))
                temp[i, j] = int(st.get("temp", 0))
                duration[i, j] = int(st.get("duration", 0))
                pipette[i, j] = int(st.get("pipette", PIP_NONE))
                rpm[i, j] = int(st.get("rpm", 0))

        return cls(
            op=op.to(dev),
            src=src.to(dev),
            dst=dst.to(dev),
            vol=vol.to(dev),
            temp=temp.to(dev),
            duration=duration.to(dev),
            pipette=pipette.to(dev),
            rpm=rpm.to(dev),
            pad_mask=pad_mask.to(dev),
        )


def make_pcr_protocol(
    device: str | torch.device | None = None,
) -> tuple[LabStateBatch, ProtocolBatch]:
    """Create a 1-protocol batch mirroring LabLean's PCR master-mix example.

    Containers:
      0: WATER (1000 uL, roomTemp, reagent bit 0)
      1: BUF   (100 uL, frozen, reagent bit 1)
      2: DNTP  (50 uL, frozen, reagent bit 2)
      3: TAQ   (10 uL, onIce/thawed, reagent bit 3)
      4: FWD   (50 uL, frozen, reagent bit 4)
      5: REV   (50 uL, frozen, reagent bit 5)
      6: TMPL  (20 uL, frozen, reagent bit 6)
      7: MM    (0 uL, cap 1500 uL)
      8..11: wells A1..A4 (0 uL, cap 200 uL)
      12: BAL_WELL (50 uL water, cap 200 uL, for balancing single-well spin)
    """
    dev = default_device(device)
    vols = [
        ul(1000), ul(100), ul(50), ul(10), ul(50), ul(50), ul(20),
        0, 0, 0, 0, 0, ul(50),
    ]
    caps = [
        ul(1500), ul(1500), ul(1500), ul(50), ul(1500), ul(1500), ul(1500),
        ul(1500), ul(200), ul(200), ul(200), ul(200), ul(200),
    ]
    frozen = [
        False, True, True, False, True, True, True,
        False, False, False, False, False, False,
    ]
    rmask = [
        1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6,
        0, 0, 0, 0, 0, 1 << 0,
    ]
    state = LabStateBatch(
        volume=torch.tensor([vols], dtype=torch.int64, device=dev),
        capacity=torch.tensor([caps], dtype=torch.int64, device=dev),
        sealed=torch.zeros((1, len(vols)), dtype=torch.bool, device=dev),
        frozen=torch.tensor([frozen], dtype=torch.bool, device=dev),
        reagent_mask=torch.tensor([rmask], dtype=torch.int64, device=dev),
        diluent_mask=1 << 0,
    )

    steps: list[dict[str, Any]] = []
    # Thaw frozen stocks
    for c_id in (1, 2, 4, 5, 6):
        steps.append({"op": OP_THAW, "src": c_id})

    def fresh_transfer(pip: int, s: int, d: int, v_ul: float) -> None:
        steps.append({"op": OP_PICK_TIP})
        steps.append({"op": OP_TRANSFER, "pipette": pip, "src": s, "dst": d, "vol": ul(v_ul)})
        steps.append({"op": OP_DROP_TIP})

    # Master mix for 5 reactions (4 + overage)
    fresh_transfer(PIP_P200, 0, 7, 184.0)
    fresh_transfer(PIP_P200, 1, 7, 25.0)
    fresh_transfer(PIP_P20, 2, 7, 5.0)
    fresh_transfer(PIP_P2, 3, 7, 1.0)

    # Mix MM and dispense 43 uL into 4 empty wells with one tip (allowed by TipClean!)
    steps.append({"op": OP_PICK_TIP})
    steps.append({"op": OP_MIX, "pipette": PIP_P200, "src": 7, "vol": ul(100.0)})
    for w in (8, 9, 10, 11):
        steps.append({"op": OP_TRANSFER, "pipette": PIP_P200, "src": 7, "dst": w, "vol": ul(43.0)})
    steps.append({"op": OP_DROP_TIP})

    # Add primers and template per well with fresh tips
    for w in (8, 9, 10, 11):
        fresh_transfer(PIP_P20, 4, w, 2.5)
        fresh_transfer(PIP_P20, 5, w, 2.5)
        steps.append({"op": OP_PICK_TIP})
        steps.append({"op": OP_TRANSFER, "pipette": PIP_P20, "src": 6, "dst": w, "vol": ul(2.0)})
        steps.append({"op": OP_MIX, "pipette": PIP_P20, "src": w, "vol": ul(10.0)})
        steps.append({"op": OP_DROP_TIP})

    # Seal well A1 and balance well (both hold 50.0 uL), spin, unseal or incubate
    steps.append({"op": OP_SEAL, "src": 8})
    steps.append({"op": OP_SEAL, "src": 12})
    steps.append({"op": OP_CENTRIFUGE, "src": 8, "dst": 12, "rpm": 1000, "duration": 30})
    steps.append({"op": OP_INCUBATE, "src": 8, "temp": 98, "duration": 30})

    proto = ProtocolBatch.from_steps_list([steps], device=dev)
    return state, proto


def make_random_valid_protocols(
    batch_size: int,
    max_steps: int = 24,
    num_containers: int = 8,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> tuple[LabStateBatch, ProtocolBatch]:
    """Generate B diverse valid protocols exercising all ops (thaw, tip, transfer, mix, incubate, seal/unseal, centrifuge)."""
    if num_containers < 6:
        raise ValueError("num_containers must be >= 6")
    if max_steps < 12:
        raise ValueError("max_steps must be >= 12")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Containers:
    # 0, 1, 2: reagent stock tubes (0 thawed, 1 & 2 frozen), high volume (8000 = 800 uL), cap 15000
    # 3..C-2: reaction wells (start 0 uL, cap 3000 = 300 uL)
    # C-1: balance tube (starts with 0 uL, cap 3000 = 300 uL)
    volume = torch.zeros((batch_size, num_containers), dtype=torch.int64)
    capacity = torch.full((batch_size, num_containers), 3000, dtype=torch.int64)
    sealed = torch.zeros((batch_size, num_containers), dtype=torch.bool)
    frozen = torch.zeros((batch_size, num_containers), dtype=torch.bool)
    reagent_mask = torch.zeros((batch_size, num_containers), dtype=torch.int64)

    for s_idx in range(3):
        volume[:, s_idx] = 8000
        capacity[:, s_idx] = 15000
        reagent_mask[:, s_idx] = 1 << s_idx
    frozen[:, 1] = True
    frozen[:, 2] = True

    op = torch.zeros((batch_size, max_steps), dtype=torch.int64)
    src = torch.zeros((batch_size, max_steps), dtype=torch.int64)
    dst = torch.zeros((batch_size, max_steps), dtype=torch.int64)
    vol = torch.zeros((batch_size, max_steps), dtype=torch.int64)
    temp = torch.zeros((batch_size, max_steps), dtype=torch.int64)
    duration = torch.zeros((batch_size, max_steps), dtype=torch.int64)
    pipette = torch.zeros((batch_size, max_steps), dtype=torch.int64)
    rpm = torch.zeros((batch_size, max_steps), dtype=torch.int64)
    pad_mask = torch.ones((batch_size, max_steps), dtype=torch.bool)

    # Construct structured valid protocols vectorized over B!
    # Step 0: thaw container 1; Step 1: thaw container 2
    op[:, 0] = OP_THAW
    src[:, 0] = 1
    pad_mask[:, 0] = False

    op[:, 1] = OP_THAW
    src[:, 1] = 2
    pad_mask[:, 1] = False

    # K transfer/mix blocks of 3 steps (pick_tip, transfer, drop_tip)
    # Leave 5 slots at the end for incubate + seal + seal + centrifuge + unseal
    usable_slots = max_steps - 2 - 5
    num_transfers = max(1, usable_slots // 3)
    cur = 2

    tracked_vol = volume.clone()
    for k in range(num_transfers):
        s_choices = torch.randint(0, 3, (batch_size,), generator=g)
        if k == 0:
            # Ensure well 3 and balance well (C-1) get equal volume in first two transfers if possible
            d_choices = torch.full((batch_size,), 3, dtype=torch.int64)
        elif k == 1:
            d_choices = torch.full((batch_size,), num_containers - 1, dtype=torch.int64)
        else:
            d_choices = torch.randint(3, num_containers - 1, (batch_size,), generator=g)

        if k in (0, 1):
            # Matched volume so well 3 and well C-1 can be centrifuged later if needed, or we balance explicitly
            v_choices = torch.full((batch_size,), 250, dtype=torch.int64)  # 25.0 uL -> P200
        else:
            v_choices = torch.randint(20, 200, (batch_size,), generator=g)  # 2.0..20.0 uL -> P20

        pip_choices = torch.where(v_choices <= 200, PIP_P20, PIP_P200)

        # Pick tip
        op[:, cur] = OP_PICK_TIP
        pad_mask[:, cur] = False
        # Transfer
        op[:, cur + 1] = OP_TRANSFER
        src[:, cur + 1] = s_choices
        dst[:, cur + 1] = d_choices
        vol[:, cur + 1] = v_choices
        pipette[:, cur + 1] = pip_choices
        pad_mask[:, cur + 1] = False
        # Drop tip
        op[:, cur + 2] = OP_DROP_TIP
        pad_mask[:, cur + 2] = False

        tracked_vol.scatter_add_(1, d_choices.unsqueeze(1), v_choices.unsqueeze(1))
        tracked_vol.scatter_add_(1, s_choices.unsqueeze(1), -v_choices.unsqueeze(1))
        cur += 3

    # Seal container 3, incubate (temperatures up to 95 C require sealing >= 80 C), and unseal
    op[:, cur] = OP_SEAL
    src[:, cur] = 3
    pad_mask[:, cur] = False
    cur += 1

    op[:, cur] = OP_INCUBATE
    src[:, cur] = 3
    temp[:, cur] = torch.randint(25, 96, (batch_size,), generator=g)
    duration[:, cur] = torch.randint(15, 180, (batch_size,), generator=g)
    pad_mask[:, cur] = False
    cur += 1

    op[:, cur] = OP_UNSEAL
    src[:, cur] = 3
    pad_mask[:, cur] = False
    cur += 1

    # Variable length: on ~70% of protocols where well 3 and C-1 are within rotor tolerance,
    # append seal + seal + centrifuge; the rest stay padded
    balanced = (tracked_vol[:, 3] - tracked_vol[:, num_containers - 1]).abs() <= ROTOR_TOL_TENTHS_UL
    do_spin = balanced & (torch.rand((batch_size,), generator=g) > 0.3)
    if cur + 3 <= max_steps:
        op[do_spin, cur] = OP_SEAL
        src[do_spin, cur] = 3
        pad_mask[do_spin, cur] = False

        op[do_spin, cur + 1] = OP_SEAL
        src[do_spin, cur + 1] = num_containers - 1
        pad_mask[do_spin, cur + 1] = False

        op[do_spin, cur + 2] = OP_CENTRIFUGE
        src[do_spin, cur + 2] = 3
        dst[do_spin, cur + 2] = num_containers - 1
        rpm[do_spin, cur + 2] = torch.randint(500, 12000, (int(do_spin.sum()),), generator=g)
        duration[do_spin, cur + 2] = 30
        pad_mask[do_spin, cur + 2] = False

    dev = default_device(device)
    state = LabStateBatch(
        volume=volume,
        capacity=capacity,
        sealed=sealed,
        frozen=frozen,
        reagent_mask=reagent_mask,
    ).to(dev)
    proto = ProtocolBatch(
        op=op,
        src=src,
        dst=dst,
        vol=vol,
        temp=temp,
        duration=duration,
        pipette=pipette,
        rpm=rpm,
        pad_mask=pad_mask,
    ).to(dev)
    return state, proto


def make_random_protocols(
    batch_size: int,
    max_steps: int = 24,
    num_containers: int = 8,
    error_prob: float = 0.5,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> tuple[LabStateBatch, ProtocolBatch]:
    """Generate a mix of valid protocols and random/corrupted protocols for differential testing."""
    state, proto = make_random_valid_protocols(
        batch_size=batch_size,
        max_steps=max_steps,
        num_containers=num_containers,
        seed=seed,
        device="cpu",
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 99991)

    corrupt_mask = torch.rand((batch_size,), generator=g) < error_prob
    n_bad = int(corrupt_mask.sum())
    if n_bad > 0:
        idx = corrupt_mask.nonzero(as_tuple=False).squeeze(1)
        # Inject completely random ops/args at 2 step positions per selected protocol (may coincide)
        for _ in range(2):
            step_pos = torch.randint(0, max_steps, (n_bad,), generator=g)
            proto.pad_mask[idx, step_pos] = False
            proto.op[idx, step_pos] = torch.randint(0, 11, (n_bad,), generator=g)
            proto.src[idx, step_pos] = torch.randint(-1, num_containers + 1, (n_bad,), generator=g)
            proto.dst[idx, step_pos] = torch.randint(-1, num_containers + 1, (n_bad,), generator=g)
            proto.vol[idx, step_pos] = torch.randint(-10, 12000, (n_bad,), generator=g)
            proto.pipette[idx, step_pos] = torch.randint(0, 6, (n_bad,), generator=g)
            proto.temp[idx, step_pos] = torch.randint(-20, 130, (n_bad,), generator=g)
            proto.duration[idx, step_pos] = torch.randint(-5, 120, (n_bad,), generator=g)
            proto.rpm[idx, step_pos] = torch.randint(-100, 20000, (n_bad,), generator=g)

    dev = default_device(device)
    return state.to(dev), proto.to(dev)
