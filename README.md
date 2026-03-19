# omniscore

`omniscore` is an installable Python package for text scoring, designed in the same one-call style as packages like BERTScore but backed by your own Hugging Face-hosted regression models.

It adds:

- A Hugging Face-native `OmniScoreConfig` and `OmniScoreModel`
- A high-level `OmniScorer` / `score(...)` API for single examples or batches
- A CLI for local files or one-off strings
- Checkpoint compatibility with `save_pretrained(...)` / `from_pretrained(...)`
- No `trust_remote_code=True` requirement once the package is installed

## Install

```bash
python3.12 -m pip install -e .
```

## Quickstart

```python
from omniscore import score

result = score(
    predictions=["A generated summary."],
    references=["A gold summary."],
    sources=["The source document."],
    model_name_or_path="your-org/omniscore-base",
)

print(result.to_list())
print(result.mean())
```

Or keep the model loaded:

```python
from omniscore import OmniScorer

scorer = OmniScorer("your-org/omniscore-base")
result = scorer.score(
    predictions=["Candidate 1", "Candidate 2"],
    references=["Reference 1", "Reference 2"],
    sources=["Source 1", "Source 2"],
)
```

## CLI

Single example:

```bash
omniscore \
  --model your-org/omniscore-base \
  --prediction "A generated summary." \
  --reference "A gold summary." \
  --source "The source document." \
  --pretty
```

Batch files:

```bash
omniscore \
  --model your-org/omniscore-base \
  --predictions-file predictions.txt \
  --references-file references.txt \
  --sources-file sources.txt \
  --pretty
```

## Hosting Models On Hugging Face

The package is built around the standard Transformers save/load flow:

```python
from transformers import AutoTokenizer
from omniscore import OmniScoreModel

model = OmniScoreModel.from_backbone(
    "distilroberta-base",
    score_names=["quality", "faithfulness"],
    source_prefix="Document:",
    reference_prefix="Reference:",
    prediction_prefix="Summary:",
)

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

model.save_pretrained("omniscore-checkpoint")
tokenizer.save_pretrained("omniscore-checkpoint")
```

Then upload that folder to Hugging Face using `transformers` / `huggingface_hub`:

```python
model.push_to_hub("your-org/omniscore-base")
tokenizer.push_to_hub("your-org/omniscore-base")
```

Hosted checkpoints can then be loaded with:

```python
from omniscore import OmniScorer

scorer = OmniScorer("your-org/omniscore-base")
```

## Package Layout

- `pyproject.toml`
- `omniscore/configuration_omniscore.py`
- `omniscore/modeling_omniscore.py`
- `omniscore/scorer.py`
- `tests/test_omniscore.py`
