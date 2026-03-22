"""Known model metadata and example inputs for omniscore."""

from __future__ import annotations

from dataclasses import dataclass

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

    def as_kwargs(self) -> dict[str, str]:
        payload = {"prediction": self.prediction}
        if self.task is not None:
            payload["tasks"] = self.task
        if self.source is not None:
            payload["source"] = self.source
        if self.reference is not None:
            payload["reference"] = self.reference
        return payload


@dataclass(frozen=True)
class KnownModel:
    """Metadata for a supported or documented hosted model."""

    repo_id: str
    family: str
    description: str
    input_format: InputFormat
    example: ModelExample


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
