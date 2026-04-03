"""Public package interface for omniscore."""

__version__ = "0.1.0"

from omniscore._auto import register_auto_classes
from omniscore.configuration_omniscore import OmniScoreConfig
from omniscore.examples import (
    KnownModel,
    ModelExample,
    get_example,
    get_known_model,
    iter_known_models,
    list_known_models,
)
from omniscore.modeling_omniscore import OmniScoreModel, OmniScoreOutput
from omniscore.scorer import OmniScoreResult, OmniScorer, score, score_example

register_auto_classes()

__all__ = [
    "OmniScoreConfig",
    "KnownModel",
    "ModelExample",
    "OmniScoreModel",
    "OmniScoreOutput",
    "OmniScoreResult",
    "OmniScorer",
    "get_example",
    "get_known_model",
    "iter_known_models",
    "list_known_models",
    "score",
    "score_example",
]
