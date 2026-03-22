"""Input formatting helpers for OmniScore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class InputFormat:
    """Formatting metadata used to flatten text-evaluation inputs."""

    task_prefix: str | None = "Task:"
    source_prefix: str | None = "Source:"
    reference_prefix: str | None = "Reference:"
    prediction_prefix: str | None = "Prediction:"
    separator: str = "\n"


def format_example(
    prediction: str,
    reference: str | None = None,
    source: str | None = None,
    task: str | None = None,
    *,
    input_format: InputFormat,
) -> str:
    """Format task/source/reference/prediction values into a single model input string."""
    segments: list[str] = []

    if task:
        segments.append(_format_segment(input_format.task_prefix, task))
    if source:
        segments.append(_format_segment(input_format.source_prefix, source))
    if reference:
        segments.append(_format_segment(input_format.reference_prefix, reference))

    segments.append(_format_segment(input_format.prediction_prefix, prediction))
    return input_format.separator.join(segments)


def _format_segment(prefix: str | None, text: str) -> str:
    normalized = text.strip()
    return f"{prefix} {normalized}".strip() if prefix else normalized


def ensure_batch(values: str | Sequence[str] | None, length: int | None = None) -> list[str | None]:
    """Normalize scalar-or-batch text inputs and broadcast scalars where needed."""
    if values is None:
        if length is None:
            return []
        return [None] * length

    if isinstance(values, str):
        if length is None:
            return [values]
        return [values] * length

    batch = list(values)
    if length is None:
        return batch
    if len(batch) != length:
        raise ValueError(f"Expected {length} items but received {len(batch)}.")
    return batch
