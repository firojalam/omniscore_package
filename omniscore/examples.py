"""Known model metadata and example inputs for omniscore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omniscore.formatting import InputFormat


@dataclass(frozen=True)
class ModelExample:
    """A documented example input for a hosted model."""

    repo_id: str
    description: str
    prediction: str
    task: str | None = None
    source: str | None = None
    reference: str | None = None

    def as_score_kwargs(self) -> dict[str, str]:
        """Return kwargs matching ``score(...)`` and ``OmniScorer.score(...)``."""
        payload = {"predictions": self.prediction}
        if self.task is not None:
            payload["tasks"] = self.task
        if self.source is not None:
            payload["sources"] = self.source
        if self.reference is not None:
            payload["references"] = self.reference
        return payload

    def as_kwargs(self) -> dict[str, str]:
        """Backward-compatible alias for score kwargs."""
        return self.as_score_kwargs()

    def to_dict(self) -> dict[str, str | None]:
        return {
            "repo_id": self.repo_id,
            "description": self.description,
            "prediction": self.prediction,
            "task": self.task,
            "source": self.source,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class KnownModel:
    """Metadata for a supported or documented hosted model."""

    repo_id: str
    family: str
    description: str
    input_format: InputFormat
    example: ModelExample
    tasks: tuple[str, ...] = ()
    model_card_url: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "family": self.family,
            "description": self.description,
            "tasks": list(self.tasks),
            "model_card_url": self.model_card_url,
            "tags": list(self.tags),
            "input_format": _input_format_to_dict(self.input_format),
            "example": self.example.to_dict(),
        }


KNOWN_MODELS: dict[str, KnownModel] = {
    "QCRI/OmniScore-deberta-v3": KnownModel(
        repo_id="QCRI/OmniScore-deberta-v3",
        family="score_predictor",
        description="Legacy DeBERTa-v3 OmniScore checkpoint published with custom score_predictor remote code.",
        input_format=InputFormat(
            task_prefix="Task:",
            source_prefix="Source:",
            reference_prefix="Reference:",
            prediction_prefix="Candidate:",
            separator="\n",
        ),
        example=ModelExample(
            repo_id="QCRI/OmniScore-deberta-v3",
            description="Headline evaluation example derived from the Hugging Face model card.",
            task="headline_evaluation",
            source="Full article text goes here.",
            prediction="Microsoft releases detailed model documentation.",
        ),
        tasks=("headline_evaluation",),
        model_card_url="https://huggingface.co/QCRI/OmniScore-deberta-v3",
        tags=("legacy", "deberta-v3", "headline-evaluation"),
    ),
}


def get_known_model(repo_id: str) -> KnownModel | None:
    """Return metadata for a known hosted model if one is registered."""
    return KNOWN_MODELS.get(repo_id)


def get_example(repo_id: str) -> ModelExample | None:
    """Return a documented example input for a known hosted model."""
    model = get_known_model(repo_id)
    return None if model is None else model.example


def list_known_models() -> tuple[str, ...]:
    """Return known model repo ids."""
    return tuple(sorted(KNOWN_MODELS))


def iter_known_models() -> tuple[KnownModel, ...]:
    """Return known model metadata objects in stable order."""
    return tuple(KNOWN_MODELS[repo_id] for repo_id in list_known_models())


def _input_format_to_dict(input_format: InputFormat) -> dict[str, str | None]:
    return {
        "task_prefix": input_format.task_prefix,
        "source_prefix": input_format.source_prefix,
        "reference_prefix": input_format.reference_prefix,
        "prediction_prefix": input_format.prediction_prefix,
        "separator": input_format.separator,
    }
