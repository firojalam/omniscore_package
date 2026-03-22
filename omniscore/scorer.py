"""High-level scoring API for omniscore."""

from __future__ import annotations

import json
import importlib.util
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from omniscore.configuration_omniscore import OmniScoreConfig
from omniscore.examples import get_known_model
from omniscore.formatting import InputFormat, ensure_batch, format_example
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
        allow_remote_code: bool = True,
        torch_dtype: str | torch.dtype | None = "auto",
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name_or_path = self._resolve_model_name(model_name_or_path)
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.allow_remote_code = allow_remote_code

        config_dict = self._load_config_dict(
            self.model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )
        self.model_family = self._infer_model_family(config_dict)
        self.known_model = get_known_model(self.model_name_or_path)
        self.input_format = self._resolve_input_format(config_dict)

        tokenizer_kwargs = dict(tokenizer_kwargs or {})
        if self.model_family == "score_predictor":
            if not self.allow_remote_code:
                raise ValueError(
                    "Model family 'score_predictor' requires remote code. "
                    "Pass allow_remote_code=True to load checkpoints such as "
                    "'QCRI/OmniScore-deberta-v3'."
                )
            tokenizer_kwargs.setdefault("trust_remote_code", True)
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
        if self.model_family == "omniscore":
            self.model = OmniScoreModel.from_pretrained(
                self.model_name_or_path,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )
        elif self.model_family == "score_predictor":
            support_path = self._prepare_legacy_code_path(
                self.model_name_or_path,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
            )
            with self._temporary_sys_path(support_path):
                config_class, model_class = self._load_legacy_score_predictor_classes(support_path)
                model_kwargs.pop("trust_remote_code", None)
                config = config_class.from_pretrained(
                    self.model_name_or_path,
                    cache_dir=cache_dir,
                    revision=revision,
                    token=token,
                )
                self.model = model_class.from_pretrained(
                    self.model_name_or_path,
                    config=config,
                    cache_dir=cache_dir,
                    revision=revision,
                    token=token,
                    torch_dtype=torch_dtype,
                    **model_kwargs,
                )
        else:
            raise ValueError(
                f"Unsupported model_type '{config_dict.get('model_type')}'. "
                "omniscore currently supports native 'omniscore' checkpoints and "
                "legacy 'score_predictor' checkpoints."
            )
        self.model.to(self.device)
        self.model.eval()

    @property
    def config(self) -> Any:
        return self.model.config

    def score(
        self,
        predictions: str | Sequence[str],
        *,
        references: str | Sequence[str] | None = None,
        sources: str | Sequence[str] | None = None,
        tasks: str | Sequence[str] | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> OmniScoreResult:
        pred_batch = ensure_batch(predictions)
        if not pred_batch:
            raise ValueError("predictions must contain at least one example.")

        ref_batch = ensure_batch(references, len(pred_batch))
        source_batch = ensure_batch(sources, len(pred_batch))
        task_batch = ensure_batch(tasks, len(pred_batch))
        text_batch = [
            format_example(
                prediction,
                reference=reference,
                source=source,
                task=task,
                input_format=self.input_format,
            )
            for prediction, reference, source, task in zip(
                pred_batch,
                ref_batch,
                source_batch,
                task_batch,
                strict=False,
            )
        ]

        current_batch_size = batch_size or self.batch_size
        current_max_length = max_length or self.max_length
        score_batches: list[np.ndarray] = []

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
                model_outputs = self.model(**encoded)
                batch_scores = self._extract_scores(model_outputs)

            score_batches.append(batch_scores.detach().float().cpu().numpy())

        stacked = np.concatenate(score_batches, axis=0)
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

    @staticmethod
    def _infer_model_family(config_dict: dict[str, Any]) -> str:
        model_type = config_dict.get("model_type")
        if model_type == OmniScoreConfig.model_type:
            return "omniscore"
        if model_type == "score_predictor":
            return "score_predictor"
        return "unknown"

    def _resolve_input_format(self, config_dict: dict[str, Any]) -> InputFormat:
        if self.known_model is not None:
            return self.known_model.input_format

        if self.model_family == "omniscore":
            return InputFormat(
                task_prefix=config_dict.get("task_prefix", "Task:"),
                source_prefix=config_dict.get("source_prefix", "Source:"),
                reference_prefix=config_dict.get("reference_prefix", "Reference:"),
                prediction_prefix=config_dict.get("prediction_prefix", "Prediction:"),
                separator=config_dict.get("separator", "\n"),
            )

        if self.model_family == "score_predictor":
            return InputFormat(
                task_prefix="Task:",
                source_prefix="Source:",
                reference_prefix="Reference:",
                prediction_prefix="Candidate:",
                separator="\n",
            )

        return InputFormat()

    @staticmethod
    def _extract_scores(outputs: Any) -> torch.Tensor:
        for attribute in ("scores", "predictions", "logits"):
            value = getattr(outputs, attribute, None)
            if value is not None:
                return value
        if isinstance(outputs, tuple) and outputs:
            return outputs[0]
        raise ValueError("Model outputs do not expose scores, predictions, or logits.")

    @staticmethod
    def _load_config_dict(
        model_name_or_path: str,
        *,
        cache_dir: str | None,
        revision: str | None,
        token: str | bool | None,
    ) -> dict[str, Any]:
        config_path = Path(model_name_or_path) / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))

        downloaded = hf_hub_download(
            repo_id=model_name_or_path,
            filename="config.json",
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))

    @staticmethod
    def _prepare_legacy_code_path(
        model_name_or_path: str,
        *,
        cache_dir: str | None,
        revision: str | None,
        token: str | bool | None,
    ) -> Path:
        local_dir = Path(model_name_or_path)
        if local_dir.is_dir():
            return local_dir

        code_dir: Path | None = None
        for filename in ("configuration_score_predictor.py", "modeling_score_predictor.py"):
            downloaded = hf_hub_download(
                repo_id=model_name_or_path,
                filename=filename,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
            )
            code_dir = Path(downloaded).parent

        assert code_dir is not None
        return code_dir

    @staticmethod
    def _load_legacy_score_predictor_classes(code_dir: Path):
        config_module = OmniScorer._load_module(
            "configuration_score_predictor",
            code_dir / "configuration_score_predictor.py",
        )
        model_module = OmniScorer._load_module(
            "modeling_score_predictor",
            code_dir / "modeling_score_predictor.py",
        )
        return config_module.ScorePredictorConfig, model_module.ScorePredictorModel

    @staticmethod
    def _load_module(module_name: str, module_path: Path):
        existing = sys.modules.get(module_name)
        if existing is not None and getattr(existing, "__file__", None) == str(module_path):
            return existing

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module {module_name} from {module_path}.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    @contextmanager
    def _temporary_sys_path(path: Path):
        path_str = str(path)
        inserted = False
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted = True
        try:
            yield
        finally:
            if inserted and path_str in sys.path:
                sys.path.remove(path_str)


def score(
    predictions: str | Sequence[str],
    *,
    references: str | Sequence[str] | None = None,
    sources: str | Sequence[str] | None = None,
    tasks: str | Sequence[str] | None = None,
    model_name_or_path: str | None = None,
    device: str = "auto",
    max_length: int = 512,
    batch_size: int = 16,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    allow_remote_code: bool = True,
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
        allow_remote_code=allow_remote_code,
        torch_dtype=torch_dtype,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    return scorer.score(
        predictions,
        references=references,
        sources=sources,
        tasks=tasks,
    )
