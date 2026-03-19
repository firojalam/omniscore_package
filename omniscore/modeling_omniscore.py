"""Modeling code for OmniScore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from omniscore.configuration_omniscore import OmniScoreConfig


@dataclass
class OmniScoreOutput(ModelOutput):
    """Standard model output for OmniScore."""

    loss: torch.FloatTensor | None = None
    scores: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class OmniScoreModel(PreTrainedModel):
    """Multi-output regression model that wraps a transformer backbone."""

    config_class = OmniScoreConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    ENCODER_MODEL_TYPES = {
        "albert",
        "bert",
        "camembert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "funnel",
        "longformer",
        "roberta",
        "xlm",
        "xlm-roberta",
    }

    DECODER_MODEL_TYPES = {
        "bloom",
        "codegen",
        "falcon",
        "gemma",
        "gemma2",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "gptj",
        "llama",
        "mistral",
        "opt",
        "phi",
        "phi3",
        "qwen",
        "qwen2",
        "starcoder",
    }

    def __init__(self, config: OmniScoreConfig) -> None:
        super().__init__(config)

        backbone_config = self._load_backbone_config(config)
        self.backbone = AutoModel.from_config(backbone_config)

        hidden_size = config.hidden_size or self._infer_hidden_size(backbone_config)
        if hidden_size is None:
            raise ValueError("Could not infer backbone hidden size. Set hidden_size in OmniScoreConfig.")

        self.config.hidden_size = hidden_size
        self.config.backbone_config = backbone_config.to_dict()
        self.config.is_encoder = (
            config.is_encoder
            if config.is_encoder is not None
            else self._detect_is_encoder(backbone_config)
        )
        self.config.pooling_strategy = (
            config.pooling_strategy
            if config.pooling_strategy is not None
            else ("cls" if self.config.is_encoder else "last_token")
        )

        self.score_head = nn.Linear(hidden_size, config.num_scores)
        self.loss_fn = nn.MSELoss()

        self.post_init()

    @classmethod
    def from_backbone(
        cls,
        backbone_model_name_or_path: str,
        *,
        score_names: list[str] | None = None,
        num_scores: int | None = None,
        minimum_score: float = 1.0,
        maximum_score: float = 5.0,
        pooling_strategy: str | None = None,
        source_prefix: str = "Source:",
        reference_prefix: str = "Reference:",
        prediction_prefix: str = "Prediction:",
        separator: str = "\n",
        **kwargs: Any,
    ) -> "OmniScoreModel":
        """Create a new OmniScore model initialized from a pretrained backbone."""
        backbone_config = AutoConfig.from_pretrained(backbone_model_name_or_path, **kwargs)
        if score_names is None:
            inferred_num_scores = num_scores or 1
            if inferred_num_scores == 1:
                score_names = ["overall"]
            else:
                score_names = [f"score_{index}" for index in range(1, inferred_num_scores + 1)]
        num_scores = num_scores or len(score_names)

        config = OmniScoreConfig(
            backbone_model_name=backbone_model_name_or_path,
            backbone_config=backbone_config.to_dict(),
            num_scores=num_scores,
            score_names=score_names,
            minimum_score=minimum_score,
            maximum_score=maximum_score,
            is_encoder=cls._detect_is_encoder(backbone_config),
            pooling_strategy=pooling_strategy,
            hidden_size=cls._infer_hidden_size(backbone_config),
            source_prefix=source_prefix,
            reference_prefix=reference_prefix,
            prediction_prefix=prediction_prefix,
            separator=separator,
        )
        model = cls(config)
        model.backbone = AutoModel.from_pretrained(backbone_model_name_or_path, config=backbone_config, **kwargs)
        model.config.backbone_config = backbone_config.to_dict()
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> OmniScoreOutput | tuple[torch.Tensor, ...]:
        """Run the backbone and predict one or more scores per example."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )

        pooled = self._pool_hidden_states(backbone_outputs.last_hidden_state, attention_mask)
        raw_scores = self.score_head(pooled)
        score_range = self.config.maximum_score - self.config.minimum_score
        scores = self.config.minimum_score + score_range * torch.sigmoid(raw_scores)

        loss = None
        if labels is not None:
            if labels.shape[-1] != self.config.num_scores:
                raise ValueError(
                    f"Expected labels with {self.config.num_scores} scores, got {labels.shape[-1]}."
                )
            loss = self.loss_fn(scores, labels.float())

        if not return_dict:
            output = (scores,)
            if output_hidden_states:
                output += (backbone_outputs.hidden_states,)
            if output_attentions:
                output += (backbone_outputs.attentions,)
            return ((loss,) + output) if loss is not None else output

        return OmniScoreOutput(
            loss=loss,
            scores=scores,
            hidden_states=backbone_outputs.hidden_states if output_hidden_states else None,
            attentions=backbone_outputs.attentions if output_attentions else None,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int | None = None) -> nn.Module:
        return self.backbone.resize_token_embeddings(
            new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    @staticmethod
    def _load_backbone_config(config: OmniScoreConfig):
        if config.backbone_config is not None:
            backbone_kwargs = dict(config.backbone_config)
            model_type = backbone_kwargs.pop("model_type")
            return AutoConfig.for_model(model_type, **backbone_kwargs)
        if config.backbone_model_name:
            return AutoConfig.from_pretrained(config.backbone_model_name)
        raise ValueError("OmniScoreConfig requires backbone_config or backbone_model_name.")

    @staticmethod
    def _infer_hidden_size(backbone_config) -> int | None:
        for attribute in ("hidden_size", "d_model", "n_embd", "dim"):
            value = getattr(backbone_config, attribute, None)
            if value is not None:
                return int(value)
        return None

    @classmethod
    def _detect_is_encoder(cls, backbone_config) -> bool:
        model_type = getattr(backbone_config, "model_type", "").lower()
        if model_type in cls.ENCODER_MODEL_TYPES:
            return True
        if model_type in cls.DECODER_MODEL_TYPES:
            return False
        if getattr(backbone_config, "is_encoder_decoder", False):
            return True
        if hasattr(backbone_config, "is_decoder"):
            return not bool(backbone_config.is_decoder)
        return True

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        strategy = self.config.pooling_strategy
        if strategy == "cls":
            return hidden_states[:, 0, :]
        if strategy == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            total = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return total / counts
        if strategy == "last_token":
            positions = attention_mask.sum(dim=1).sub(1).clamp(min=0).long()
            batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_index, positions]
        raise ValueError(f"Unsupported pooling strategy: {strategy}")
