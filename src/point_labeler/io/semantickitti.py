"""SemanticKITTI-compatible point cloud and label I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

PointMode = Literal["xyz", "xyzi"]

_FLOATS_PER_POINT = 4


def read_semantickitti_bin(
    file_path: str | Path,
    mode: PointMode = "xyzi",
) -> NDArray[np.float32]:
    """Read SemanticKITTI .bin point cloud.

    The file layout is float32 Nx4 (x, y, z, intensity).
    Use mode='xyz' to return Nx3 and mode='xyzi' to return Nx4.
    """
    path = Path(file_path)
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % _FLOATS_PER_POINT != 0:
        raise ValueError(
            f"Invalid SemanticKITTI bin file: {path}. "
            f"float count {raw.size} is not divisible by 4."
        )

    points_xyzi = raw.reshape((-1, _FLOATS_PER_POINT))
    if mode == "xyzi":
        return points_xyzi
    if mode == "xyz":
        return points_xyzi[:, :3]
    raise ValueError(f"Unsupported mode: {mode}")


def write_semantickitti_bin(
    file_path: str | Path,
    points: NDArray[np.floating],
    input_mode: PointMode = "xyzi",
    default_intensity: float = 0.0,
) -> None:
    """Write points to SemanticKITTI .bin as float32 Nx4.

    - input_mode='xyzi': expects Nx4
    - input_mode='xyz': expects Nx3, and fills intensity with default_intensity
    """
    path = Path(file_path)
    arr = np.asarray(points)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    if input_mode == "xyzi":
        if arr.shape[1] != 4:
            raise ValueError(f"Expected Nx4 for xyzi, got shape {arr.shape}")
        out = arr.astype(np.float32, copy=False)
    elif input_mode == "xyz":
        if arr.shape[1] != 3:
            raise ValueError(f"Expected Nx3 for xyz, got shape {arr.shape}")
        out = np.empty((arr.shape[0], 4), dtype=np.float32)
        out[:, :3] = arr.astype(np.float32, copy=False)
        out[:, 3] = np.float32(default_intensity)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    out.tofile(path)


def read_semantickitti_label(file_path: str | Path) -> NDArray[np.uint32]:
    """Read SemanticKITTI .label as uint32 array."""
    path = Path(file_path)
    return np.fromfile(path, dtype=np.uint32)


def write_semantickitti_label(
    file_path: str | Path,
    labels: NDArray[np.unsignedinteger],
) -> None:
    """Write SemanticKITTI .label as uint32 array."""
    path = Path(file_path)
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D label array, got shape {arr.shape}")
    arr.astype(np.uint32, copy=False).tofile(path)


def split_semantickitti_label(
    labels: NDArray[np.unsignedinteger],
) -> Tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    """Split uint32 labels into semantic_id and instance_id.

    SemanticKITTI convention:
    - lower 16 bits: semantic_id
    - upper 16 bits: instance_id
    """
    arr = np.asarray(labels, dtype=np.uint32)
    semantic_id = (arr & 0xFFFF).astype(np.uint16, copy=False)
    instance_id = (arr >> 16).astype(np.uint16, copy=False)
    return semantic_id, instance_id


def pack_semantickitti_label(
    semantic_id: NDArray[np.unsignedinteger],
    instance_id: NDArray[np.unsignedinteger] | None = None,
) -> NDArray[np.uint32]:
    """Pack semantic_id and instance_id into SemanticKITTI uint32 labels."""
    sem = np.asarray(semantic_id)
    if sem.ndim != 1:
        raise ValueError(f"semantic_id must be 1D, got shape {sem.shape}")

    if instance_id is None:
        ins = np.zeros_like(sem, dtype=np.uint32)
    else:
        ins = np.asarray(instance_id)
        if ins.ndim != 1:
            raise ValueError(f"instance_id must be 1D, got shape {ins.shape}")
        if ins.shape[0] != sem.shape[0]:
            raise ValueError(
                "semantic_id and instance_id must have the same length: "
                f"{sem.shape[0]} != {ins.shape[0]}"
            )

    sem_u32 = sem.astype(np.uint32, copy=False)
    ins_u32 = ins.astype(np.uint32, copy=False)
    packed = ((ins_u32 & 0xFFFF) << 16) | (sem_u32 & 0xFFFF)
    return packed.astype(np.uint32, copy=False)

