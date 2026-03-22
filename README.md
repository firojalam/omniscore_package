# omniscore

`omniscore` is an installable Python package for text scoring, designed in the same one-call style as packages like BERTScore but backed by your own Hugging Face-hosted regression models.

It includes:

- A Hugging Face-native `OmniScoreConfig` and `OmniScoreModel`
- A high-level `OmniScorer` / `score(...)` API for single examples or batches
- A CLI for local files or one-off strings
- Checkpoint compatibility with `save_pretrained(...)` / `from_pretrained(...)`
- Native `omniscore` checkpoints without `trust_remote_code=True`
- Built-in support for legacy `score_predictor` checkpoints such as `QCRI/OmniScore-deberta-v3`

## Install

```bash
python3.12 -m pip install -e .
```

The runtime dependencies include `sentencepiece` so DeBERTa-v3 based checkpoints load cleanly.

## Quickstart

```python
from omniscore import score

result = score(
    predictions=["A generated summary."],
    references=["A gold summary."],
    sources=["The source document."],
    tasks=["summarization"],
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
    tasks=["summarization", "summarization"],
)
```

## Real Model Example

The package includes metadata for known hosted models and currently documents `QCRI/OmniScore-deberta-v3`.

```python
from omniscore import OmniScorer, get_example

example = get_example("QCRI/OmniScore-deberta-v3")
scorer = OmniScorer("QCRI/OmniScore-deberta-v3")

result = scorer.score(
    predictions=example.prediction,
    sources=example.source,
    references=example.reference,
    tasks=example.task,
)

print(result.to_list())
```

Equivalent explicit example, matching the Hugging Face model card:

```python
from omniscore import OmniScorer

scorer = OmniScorer("QCRI/OmniScore-deberta-v3")
result = scorer.score(
    predictions="Microsoft releases detailed model documentation.",
    sources="Full article text goes here.",
    tasks="headline_evaluation",
)

print(result.to_list())
```

`omniscore` detects that this is a legacy `score_predictor` checkpoint and internally handles the remote-code loading path for you.

## Example Notebook

A clean Colab-oriented example notebook is available at `examples/omniscore_qcri_deberta_v3_colab.ipynb`.

It walks through:

- verifying the GPU runtime
- loading `QCRI/OmniScore-deberta-v3`
- running the documented example with `OmniScorer`
- checking the returned score structure

## CLI

Single example:

```bash
omniscore \
  --model QCRI/OmniScore-deberta-v3 \
  --prediction "Microsoft releases detailed model documentation." \
  --source "Full article text goes here." \
  --task headline_evaluation \
  --pretty
```

Batch files:

```bash
omniscore \
  --model QCRI/OmniScore-deberta-v3 \
  --predictions-file predictions.txt \
  --references-file references.txt \
  --sources-file sources.txt \
  --tasks-file tasks.txt \
  --pretty
```

## Hosting Models On Hugging Face

The package is built around the standard Transformers save/load flow for native checkpoints:

```python
from transformers import AutoTokenizer
from omniscore import OmniScoreModel

model = OmniScoreModel.from_backbone(
    "distilroberta-base",
    score_names=["quality", "faithfulness"],
    task_prefix="Task:",
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

For legacy hosted checkpoints published with custom remote code, use the same API:

```python
from omniscore import OmniScorer

scorer = OmniScorer("QCRI/OmniScore-deberta-v3")
```

## Supported Families

`omniscore` currently supports:

- Native `omniscore` checkpoints saved from `OmniScoreModel`
- Legacy `score_predictor` checkpoints, including `QCRI/OmniScore-deberta-v3`

Future models can use either family. If you publish more hosted models later, users only need to switch `model_name_or_path`.

## Package Layout

- `pyproject.toml`
- `omniscore/configuration_omniscore.py`
- `omniscore/examples.py`
- `omniscore/modeling_omniscore.py`
- `omniscore/scorer.py`
- `tests/test_omniscore.py`
