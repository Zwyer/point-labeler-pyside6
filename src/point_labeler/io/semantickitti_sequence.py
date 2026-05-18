"""SemanticKITTI sequence discovery and indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class SequenceFrame:
    """A single frame entry inside a SemanticKITTI sequence."""

    frame_id: str
    bin_path: Path
    label_path: Path | None


@dataclass(frozen=True)
class SemanticKittiSequence:
    """Resolved sequence directory and its indexed frames."""

    sequence_dir: Path
    velodyne_dir: Path
    labels_dir: Path | None
    frames: List[SequenceFrame]


def resolve_sequence_dir(user_selected_dir: str | Path) -> Path:
    """Resolve a user-selected path to a SemanticKITTI sequence directory.

    Accepted examples:
    - .../sequences/00
    - .../semantic/00
    - .../sequences/00/velodyne
    """
    selected = Path(user_selected_dir).expanduser().resolve()

    if selected.name.lower() == "velodyne":
        return selected.parent
    if (selected / "velodyne").is_dir():
        return selected
    raise FileNotFoundError(
        f"Cannot resolve SemanticKITTI sequence from: {selected}. "
        "Expected this folder (or itself) to map to a sequence containing 'velodyne'."
    )


def load_semantickitti_sequence(user_selected_dir: str | Path) -> SemanticKittiSequence:
    """Load all .bin files under a SemanticKITTI sequence and match labels.

    Matching rule:
    - velodyne/<frame>.bin <-> labels/<frame>.label (same stem)
    """
    sequence_dir = resolve_sequence_dir(user_selected_dir)
    velodyne_dir = sequence_dir / "velodyne"

    labels_dir_candidate = sequence_dir / "labels"
    labels_dir = labels_dir_candidate if labels_dir_candidate.is_dir() else None

    bin_paths = sorted(velodyne_dir.glob("*.bin"))
    if not bin_paths:
        raise FileNotFoundError(f"No .bin files found in: {velodyne_dir}")

    frames: List[SequenceFrame] = []
    for bin_path in bin_paths:
        frame_id = bin_path.stem
        label_path = None
        if labels_dir is not None:
            candidate = labels_dir / f"{frame_id}.label"
            if candidate.is_file():
                label_path = candidate

        frames.append(
            SequenceFrame(
                frame_id=frame_id,
                bin_path=bin_path,
                label_path=label_path,
            )
        )

    return SemanticKittiSequence(
        sequence_dir=sequence_dir,
        velodyne_dir=velodyne_dir,
        labels_dir=labels_dir,
        frames=frames,
    )
