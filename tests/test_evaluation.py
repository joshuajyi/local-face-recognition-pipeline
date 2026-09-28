from pathlib import Path

import pytest

from face_pipeline.evaluation import (
    EvaluationRecord,
    canonical_label,
    discover_evaluation_cases,
    summarize_evaluation,
)


def record(
    expected: str,
    predicted: str | None,
    *,
    face_count: int = 1,
) -> EvaluationRecord:
    return EvaluationRecord(
        image="example.jpg",
        expected=expected,
        predicted=predicted,
        score=0.75 if predicted else None,
        face_count=face_count,
    )


def test_unknown_label_is_case_insensitive() -> None:
    assert canonical_label(" unknown ") == "Unknown"


@pytest.mark.parametrize(
    ("item", "expected_outcome"),
    [
        (record("Joshua", "Joshua"), "correct"),
        (record("Unknown", "Unknown"), "correct"),
        (record("Unknown", "Joshua"), "false_accept"),
        (record("Joshua", "Unknown"), "false_reject"),
        (record("Joshua", "Alice"), "wrong_identity"),
        (record("Joshua", None, face_count=0), "no_face"),
        (record("Joshua", None, face_count=2), "multiple_faces"),
    ],
)
def test_evaluation_outcomes(item: EvaluationRecord, expected_outcome: str) -> None:
    assert item.outcome == expected_outcome


def test_discover_evaluation_cases(tmp_path: Path) -> None:
    known = tmp_path / "Joshua"
    unknown = tmp_path / "unknown"
    known.mkdir()
    unknown.mkdir()
    (known / "known.jpg").touch()
    (unknown / "unknown.PNG").touch()
    (known / "notes.txt").touch()

    cases = discover_evaluation_cases(tmp_path)

    assert [(case.expected, case.image.name) for case in cases] == [
        ("Joshua", "known.jpg"),
        ("Unknown", "unknown.PNG"),
    ]


def test_discover_rejects_empty_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No JPG or PNG images"):
        discover_evaluation_cases(tmp_path)


def test_evaluation_summary() -> None:
    records = [
        record("Joshua", "Joshua"),
        record("Unknown", "Unknown"),
        record("Unknown", "Joshua"),
        record("Joshua", "Unknown"),
        record("Joshua", "Alice"),
        record("Joshua", None, face_count=0),
    ]

    summary = summarize_evaluation(records)

    assert summary["total_images"] == 6
    assert summary["evaluated_images"] == 5
    assert summary["correct"] == 2
    assert summary["accuracy"] == pytest.approx(0.4)
    assert summary["false_accepts"] == 1
    assert summary["false_rejects"] == 1
    assert summary["wrong_identity"] == 1
    assert summary["no_face"] == 1
