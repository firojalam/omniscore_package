"""Helpers to register omniscore with Hugging Face auto classes."""

from transformers import AutoConfig, AutoModel

from omniscore.configuration_omniscore import OmniScoreConfig
from omniscore.modeling_omniscore import OmniScoreModel


def register_auto_classes() -> None:
    """Register config/model so HF checkpoints load without remote code."""
    try:
        AutoConfig.register(OmniScoreConfig.model_type, OmniScoreConfig)
    except ValueError:
        pass

    try:
        AutoModel.register(OmniScoreConfig, OmniScoreModel)
    except ValueError:
        pass
