"""High-level scoring API for omniscore."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from omniscore.configuration_omniscore import OmniScoreConfig
from omniscore.formatting import ensure_batch, format_example
from omniscore.modeling_omniscore import OmniScoreModel


@dataclass(frozen=True)
class OmniScoreResult:
    """Structured score output returned by OmniScorer."""

    score_names: tuple[str, ...]
    scores: np.ndarray

    @property
    def overall(self) -> np.ndarray:
        return self.scores.mean(axis=1)

    def mean(self) -> dict[str, float]:
        means = self.scores.mean(axis=0)
        payload = {name: float(value) for name, value in zip(self.score_names, means, strict=False)}
        payload["overall"] = float(self.overall.mean())
        return payload

    def to_list(self) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for row in self.scores:
            payload = {
                name: float(value) for name, value in zip(self.score_names, row, strict=False)
            }
            payload["overall"] = float(row.mean())
            rows.append(payload)
        return rows

    def __len__(self) -> int:
        return int(self.scores.shape[0])


class OmniScorer:
    """Load a hosted OmniScore model and compute scores for text batches."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
        cache_dir: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        torch_dtype: str | torch.dtype | None = "auto",
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name_or_path = self._resolve_model_name(model_name_or_path)
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            fallback_pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback_pad_token is None:
                raise ValueError(
                    "Tokenizer does not define a pad_token, eos_token, or unk_token."
                )
            self.tokenizer.pad_token = fallback_pad_token

        model_kwargs = dict(model_kwargs or {})
        self.model = OmniScoreModel.from_pretrained(
            self.model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def config(self) -> OmniScoreConfig:
        return self.model.config

    def score(
        self,
        predictions: str | Sequence[str],
        *,
        references: str | Sequence[str] | None = None,
        sources: str | Sequence[str] | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> OmniScoreResult:
        pred_batch = ensure_batch(predictions)
        if not pred_batch:
            raise ValueError("predictions must contain at least one example.")

        ref_batch = ensure_batch(references, len(pred_batch))
        source_batch = ensure_batch(sources, len(pred_batch))
        text_batch = [
            format_example(prediction, reference=reference, source=source, config=self.config)
            for prediction, reference, source in zip(pred_batch, ref_batch, source_batch, strict=False)
        ]

        current_batch_size = batch_size or self.batch_size
        current_max_length = max_length or self.max_length
        outputs: list[np.ndarray] = []

        for start in range(0, len(text_batch), current_batch_size):
            chunk = text_batch[start : start + current_batch_size]
            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=current_max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                batch_scores = self.model(**encoded).scores

            outputs.append(batch_scores.detach().float().cpu().numpy())

        stacked = np.concatenate(outputs, axis=0)
        return OmniScoreResult(score_names=tuple(self.config.score_names), scores=stacked)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_model_name(model_name_or_path: str | None) -> str:
        if model_name_or_path:
            return model_name_or_path
        env_value = os.getenv("OMNISCORE_MODEL")
        if env_value:
            return env_value
        raise ValueError(
            "Pass model_name_or_path or set the OMNISCORE_MODEL environment variable."
        )


def score(
    predictions: str | Sequence[str],
    *,
    references: str | Sequence[str] | None = None,
    sources: str | Sequence[str] | None = None,
    model_name_or_path: str | None = None,
    device: str = "auto",
    max_length: int = 512,
    batch_size: int = 16,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    torch_dtype: str | torch.dtype | None = "auto",
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> OmniScoreResult:
    """Convenience function matching the one-shot style of packages like BERTScore."""
    scorer = OmniScorer(
        model_name_or_path=model_name_or_path,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        torch_dtype=torch_dtype,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    return scorer.score(
        predictions,
        references=references,
        sources=sources,
    )
