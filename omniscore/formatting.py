"""Input formatting helpers for OmniScore."""

from __future__ import annotations

from typing import Sequence

from omniscore.configuration_omniscore import OmniScoreConfig


def format_example(
    prediction: str,
    reference: str | None = None,
    source: str | None = None,
    *,
    config: OmniScoreConfig,
) -> str:
    """Format a prediction/reference/source triple into model input text."""
    segments: list[str] = []

    if source:
        segments.append(_format_segment(config.source_prefix, source))
    if reference:
        segments.append(_format_segment(config.reference_prefix, reference))

    segments.append(_format_segment(config.prediction_prefix, prediction))
    return config.separator.join(segments)


def _format_segment(prefix: str, text: str) -> str:
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
