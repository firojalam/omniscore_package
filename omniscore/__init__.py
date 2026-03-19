"""Public package interface for omniscore."""

from omniscore._auto import register_auto_classes
from omniscore.configuration_omniscore import OmniScoreConfig
from omniscore.modeling_omniscore import OmniScoreModel, OmniScoreOutput
from omniscore.scorer import OmniScoreResult, OmniScorer, score

register_auto_classes()

__all__ = [
    "OmniScoreConfig",
    "OmniScoreModel",
    "OmniScoreOutput",
    "OmniScoreResult",
    "OmniScorer",
    "score",
]
