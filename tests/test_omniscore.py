from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from omniscore import OmniScoreModel, OmniScorer, score


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
    )

    assert result.score_names == ("quality", "faithfulness")
    assert result.scores.shape == (2, 2)
    assert len(result.to_list()) == 2
    assert set(result.mean()) == {"quality", "faithfulness", "overall"}

    single = score(
        "A single summary.",
        references="A single reference.",
        sources="A single source document.",
        model_name_or_path=str(model_dir),
        device="cpu",
        max_length=64,
    )
    assert single.scores.shape == (1, 2)


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
