"""Public package interface for omniscore."""

from omniscore._auto import register_auto_classes
from omniscore.configuration_omniscore import OmniScoreConfig
from omniscore.examples import KnownModel, ModelExample, get_example, get_known_model, list_known_models
from omniscore.modeling_omniscore import OmniScoreModel, OmniScoreOutput
from omniscore.scorer import OmniScoreResult, OmniScorer, score

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
    "list_known_models",
    "score",
]
