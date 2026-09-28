"""Helpers for evaluating recognition results from a labeled image folder."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
UNKNOWN_LABEL = "Unknown"


def canonical_label(label: str) -> str:
    """Normalize the reserved Unknown label while preserving profile names."""

    clean = " ".join(label.strip().split())
    if not clean:
        raise ValueError("Evaluation labels cannot be empty")
    return UNKNOWN_LABEL if clean.casefold() == UNKNOWN_LABEL.casefold() else clean


@dataclass(frozen=True)
class EvaluationCase:
    image: Path
    expected: str


@dataclass(frozen=True)
class EvaluationRecord:
    image: str
    expected: str
    predicted: str | None
    score: float | None
    face_count: int

    @property
    def outcome(self) -> str:
        if self.face_count == 0:
            return "no_face"
        if self.face_count > 1:
            return "multiple_faces"
        if self.predicted is None:
            return "no_prediction"

        expected = canonical_label(self.expected)
        predicted = canonical_label(self.predicted)
        if predicted == expected:
            return "correct"
        if expected == UNKNOWN_LABEL:
            return "false_accept"
        if predicted == UNKNOWN_LABEL:
            return "false_reject"
        return "wrong_identity"

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["outcome"] = self.outcome
        return data


def discover_evaluation_cases(root: Path) -> list[EvaluationCase]:
    """Read images from ROOT/<expected label>/*.{jpg,jpeg,png}."""

    if not root.is_dir():
        raise ValueError(f"Evaluation directory does not exist: {root}")

    cases: list[EvaluationCase] = []
    for label_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        expected = canonical_label(label_directory.name)
        image_paths = sorted(
            path
            for path in label_directory.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
        cases.extend(EvaluationCase(image=path, expected=expected) for path in image_paths)

    if not cases:
        raise ValueError(
            f"No JPG or PNG images found. Use folders such as {root / 'Joshua'} and "
            f"{root / UNKNOWN_LABEL}."
        )
    return cases


def summarize_evaluation(records: list[EvaluationRecord]) -> dict[str, int | float]:
    outcomes = Counter(record.outcome for record in records)
    evaluated = sum(
        outcomes[key] for key in ("correct", "false_accept", "false_reject", "wrong_identity")
    )
    correct = outcomes["correct"]
    return {
        "total_images": len(records),
        "evaluated_images": evaluated,
        "correct": correct,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "false_accepts": outcomes["false_accept"],
        "false_rejects": outcomes["false_reject"],
        "wrong_identity": outcomes["wrong_identity"],
        "no_face": outcomes["no_face"],
        "multiple_faces": outcomes["multiple_faces"],
        "no_prediction": outcomes["no_prediction"],
    }
