"""Configuration for OmniScore models."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


DEFAULT_SCORE_NAMES = [
    "informativeness",
    "clarity",
    "plausibility",
    "faithfulness",
]


class OmniScoreConfig(PretrainedConfig):
    """Configuration for a multi-score regression model backed by a transformer."""

    model_type = "omniscore"

    def __init__(
        self,
        backbone_model_name: str = "",
        backbone_config: dict[str, Any] | None = None,
        num_scores: int = 4,
        score_names: list[str] | None = None,
        minimum_score: float = 1.0,
        maximum_score: float = 5.0,
        is_encoder: bool | None = None,
        pooling_strategy: str | None = None,
        hidden_size: int | None = None,
        task_prefix: str | None = "Task:",
        source_prefix: str = "Source:",
        reference_prefix: str = "Reference:",
        prediction_prefix: str = "Prediction:",
        separator: str = "\n",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if maximum_score <= minimum_score:
            raise ValueError("maximum_score must be greater than minimum_score.")

        if score_names is None:
            score_names = list(DEFAULT_SCORE_NAMES[:num_scores])
            if len(score_names) < num_scores:
                score_names.extend(
                    f"score_{index}" for index in range(len(score_names) + 1, num_scores + 1)
                )

        if len(score_names) != num_scores:
            raise ValueError("len(score_names) must match num_scores.")

        self.backbone_model_name = backbone_model_name
        self.backbone_config = backbone_config
        self.num_scores = num_scores
        self.score_names = score_names
        self.minimum_score = minimum_score
        self.maximum_score = maximum_score
        self.is_encoder = is_encoder
        self.pooling_strategy = pooling_strategy
        self.hidden_size = hidden_size
        self.task_prefix = task_prefix
        self.source_prefix = source_prefix
        self.reference_prefix = reference_prefix
        self.prediction_prefix = prediction_prefix
        self.separator = separator
