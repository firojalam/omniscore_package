from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from omniscore import (
    ModelExample,
    OmniScoreModel,
    OmniScorer,
    get_example,
    get_known_model,
    list_known_models,
    score,
    score_example,
)
from omniscore.cli import main as cli_main


def test_omniscore_roundtrip_and_scoring(tmp_path: Path) -> None:
    backbone_dir = _create_local_backbone(tmp_path / "backbone")
    model = OmniScoreModel.from_backbone(
        str(backbone_dir),
        score_names=["quality", "faithfulness"],
        source_prefix="Document:",
        reference_prefix="Reference:",
        prediction_prefix="Summary:",
    )

    model_dir = tmp_path / "omniscore-model"
    model.save_pretrained(model_dir)
    PreTrainedTokenizerFast.from_pretrained(backbone_dir).save_pretrained(model_dir)

    scorer = OmniScorer(str(model_dir), device="cpu", batch_size=2, max_length=64)
    result = scorer.score(
        predictions=["A short generated summary.", "Another summary."],
        references=["A short reference summary.", "Another reference."],
        sources=["A source document.", "A second source document."],
        tasks=["summarization", "summarization"],
    )

    assert result.score_names == ("quality", "faithfulness")
    assert result.scores.shape == (2, 2)
    assert len(result.to_list()) == 2
    assert set(result.mean()) == {"quality", "faithfulness", "overall"}

    single = score(
        "A single summary.",
        references="A single reference.",
        sources="A single source document.",
        tasks="summarization",
        model_name_or_path=str(model_dir),
        device="cpu",
        max_length=64,
    )
    assert single.scores.shape == (1, 2)


def test_known_model_example_metadata() -> None:
    known = get_known_model("QCRI/OmniScore-deberta-v3")
    assert known is not None
    assert known.family == "score_predictor"
    assert known.input_format.prediction_prefix == "Candidate:"
    assert "QCRI/OmniScore-deberta-v3" in list_known_models()

    example = get_example("QCRI/OmniScore-deberta-v3")
    assert example is not None
    assert example.task == "headline_evaluation"
    assert example.as_score_kwargs() == {
        "predictions": "Microsoft releases detailed model documentation.",
        "tasks": "headline_evaluation",
        "sources": "Full article text goes here.",
    }
    assert known.to_dict()["model_card_url"] == "https://huggingface.co/QCRI/OmniScore-deberta-v3"


def test_score_predictor_family_uses_documented_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "legacy-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
{
  "model_type": "score_predictor",
  "score_names": ["informativeness", "clarity", "plausibility", "faithfulness"]
}
        """.strip(),
        encoding="utf-8",
    )

    class FakeTokenizer:
        pad_token = "[PAD]"
        eos_token = None
        unk_token = "[UNK]"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def __call__(self, texts, return_tensors, padding, truncation, max_length):
            self.calls.append(list(texts))
            batch_size = len(texts)
            return {
                "input_ids": torch.zeros((batch_size, 4), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
            }

    class FakeLegacyConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(
                score_names=["informativeness", "clarity", "plausibility", "faithfulness"]
            )

    class FakeLegacyModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                score_names=["informativeness", "clarity", "plausibility", "faithfulness"]
            )

        @classmethod
        def from_pretrained(cls, *args, config=None, **kwargs):
            model = cls()
            if config is not None:
                model.config = config
            return model

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            return SimpleNamespace(
                predictions=torch.full((batch_size, 4), 3.5, dtype=torch.float32)
            )

    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        "omniscore.scorer.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )
    monkeypatch.setattr(
        "omniscore.scorer.OmniScorer._load_legacy_score_predictor_classes",
        staticmethod(lambda *args, **kwargs: (FakeLegacyConfig, FakeLegacyModel)),
    )

    scorer = OmniScorer(str(model_dir), device="cpu")
    result = scorer.score(
        predictions="Microsoft releases detailed model documentation.",
        sources="Full article text goes here.",
        tasks="headline_evaluation",
    )

    assert result.scores.shape == (1, 4)
    assert np.allclose(result.scores, 3.5)
    [formatted] = fake_tokenizer.calls[0]
    assert formatted == (
        "Task: headline_evaluation\n"
        "Source: Full article text goes here.\n"
        "Candidate: Microsoft releases detailed model documentation."
    )


def test_prepare_legacy_code_path_snapshots_python_support_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote_dir = tmp_path / "remote-support"
    remote_dir.mkdir()
    call_kwargs: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        call_kwargs.update(kwargs)
        return str(remote_dir)

    monkeypatch.setattr("omniscore.scorer.snapshot_download", fake_snapshot_download)

    support_dir = OmniScorer._prepare_legacy_code_path(
        "QCRI/OmniScore-deberta-v3",
        cache_dir=None,
        revision=None,
        token=None,
    )

    assert support_dir == remote_dir
    assert call_kwargs["repo_id"] == "QCRI/OmniScore-deberta-v3"
    assert call_kwargs["allow_patterns"] == ["*.py", "config.json"]


def test_load_legacy_score_predictor_classes_supports_relative_imports(tmp_path: Path) -> None:
    code_dir = tmp_path / "legacy-remote"
    code_dir.mkdir()
    (code_dir / "configuration_score_predictor.py").write_text(
        """
class ScorePredictorConfig:
    pass
        """.strip(),
        encoding="utf-8",
    )
    (code_dir / "modeling_score_predictor.py").write_text(
        """
from .configuration_score_predictor import ScorePredictorConfig


class ScorePredictorModel:
    config_class = ScorePredictorConfig
        """.strip(),
        encoding="utf-8",
    )

    config_class, model_class = OmniScorer._load_legacy_score_predictor_classes(code_dir)

    assert config_class.__name__ == "ScorePredictorConfig"
    assert model_class.__name__ == "ScorePredictorModel"
    assert model_class.config_class is config_class


def test_load_module_replaces_stale_module(tmp_path: Path) -> None:
    module_name = "_omniscore_test_stale_module"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / f"{module_name}.py"
    second_path = second_dir / f"{module_name}.py"
    first_path.write_text("VALUE = 1\n", encoding="utf-8")
    second_path.write_text("VALUE = 2\n", encoding="utf-8")

    try:
        first_module = OmniScorer._load_module(module_name, first_path)
        second_module = OmniScorer._load_module(module_name, second_path)
    finally:
        sys.modules.pop(module_name, None)

    assert first_module.VALUE == 1
    assert second_module.VALUE == 2
    assert second_module is not first_module


def test_score_example_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backbone_dir = _create_local_backbone(tmp_path / "backbone")
    model = OmniScoreModel.from_backbone(
        str(backbone_dir),
        score_names=["quality", "faithfulness"],
    )
    model_dir = tmp_path / "omniscore-model"
    model.save_pretrained(model_dir)
    PreTrainedTokenizerFast.from_pretrained(backbone_dir).save_pretrained(model_dir)

    example = ModelExample(
        repo_id="demo/model",
        description="Demo example.",
        prediction="A generated summary.",
        source="A source document.",
        reference="A reference summary.",
        task="summarization",
    )
    monkeypatch.setattr("omniscore.scorer.get_example", lambda repo_id: example)

    scorer = OmniScorer(str(model_dir), device="cpu", max_length=64)
    result = scorer.score_example(repo_id="demo/model")
    assert result.scores.shape == (1, 2)

    one_shot = score_example(
        model_name_or_path=str(model_dir),
        repo_id="demo/model",
        device="cpu",
        max_length=64,
    )
    assert one_shot.scores.shape == (1, 2)


def test_cli_lists_models(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["omniscore", "--list-models", "--pretty"])
    cli_main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"][0]["repo_id"] == "QCRI/OmniScore-deberta-v3"


def test_cli_use_example_writes_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "result.json"

    class FakeResult:
        def to_dict(self) -> dict[str, object]:
            return {
                "score_names": ["informativeness"],
                "scores": [{"informativeness": 4.2, "overall": 4.2}],
                "mean": {"informativeness": 4.2, "overall": 4.2},
            }

    class FakeScorer:
        def __init__(self, model_name_or_path: str | None = None, **kwargs: object) -> None:
            self.model_name_or_path = model_name_or_path

        def score_example(self, repo_id: str | None = None) -> FakeResult:
            assert repo_id == "QCRI/OmniScore-deberta-v3"
            return FakeResult()

    monkeypatch.setattr("omniscore.cli.OmniScorer", FakeScorer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "omniscore",
            "--model",
            "QCRI/OmniScore-deberta-v3",
            "--use-example",
            "--pretty",
            "--output-file",
            str(output_file),
        ],
    )
    cli_main()

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["model"] == "QCRI/OmniScore-deberta-v3"
    assert stdout_payload["known_model"]["family"] == "score_predictor"


@pytest.mark.integration
def test_live_qcri_model_example() -> None:
    if os.getenv("OMNISCORE_RUN_LIVE_HF") != "1":
        pytest.skip("Set OMNISCORE_RUN_LIVE_HF=1 to run the live Hugging Face integration test.")

    example = get_example("QCRI/OmniScore-deberta-v3")
    assert example is not None

    scorer = OmniScorer("QCRI/OmniScore-deberta-v3", device="cpu", max_length=128)
    result = scorer.score(
        predictions=example.prediction,
        references=example.reference,
        sources=example.source,
        tasks=example.task,
    )

    assert result.score_names == (
        "informativeness",
        "clarity",
        "plausibility",
        "faithfulness",
    )
    assert result.scores.shape == (1, 4)
    assert np.all(result.scores >= 1.0)
    assert np.all(result.scores <= 5.0)


def _create_local_backbone(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "document": 5,
        "reference": 6,
        "summary": 7,
        "source": 8,
        "a": 9,
        "short": 10,
        "generated": 11,
        "another": 12,
        "single": 13,
        ".": 14,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    fast_tokenizer.save_pretrained(path)

    backbone_config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    backbone = BertModel(backbone_config)
    backbone.save_pretrained(path)
    return path
