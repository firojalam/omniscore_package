# 🎯 omniscore

[![Python Version](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/collections/QCRI/omniscore)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1985As0_gN1mcRIt9Z_Q3jYRT2VBCzdo9)

`omniscore` is a lightweight Python package for evaluating the quality of Natural Language Generation (NLG) and generated text. Whether you are evaluating Question Answering (QA), text summarization, explanations, or LLM chat interactions, `omniscore` is designed to integrate seamlessly with [OmniScore models](https://huggingface.co/collections/QCRI/omniscore).

## ✨ Key Features

- **Hugging Face Native:** Includes custom `OmniScoreConfig` and `OmniScoreModel` classes that fit right into your existing Hugging Face workflows.
- **Easy-to-Use API:** High-level `OmniScorer` and `score(...)` API for processing single examples or large batches with minimal boilerplate.
- **Command-Line Interface (CLI):** Built-in CLI for quickly scoring local text files or one-off strings directly from your terminal.
- **Standard Serialization:** Full checkpoint compatibility using standard `save_pretrained(...)` and `from_pretrained(...)` methods.
- **Secure Loading:** Load native `omniscore` checkpoints safely without needing `trust_remote_code=True`.
- **Backward Compatibility:** Built-in support for legacy `score_predictor` checkpoints (such as `QCRI/OmniScore-deberta-v3`).

---

## 📑 Table of Contents
1. [Installation](#-installation)
2. [Quickstart: Python API](#-quickstart-python-api)
3. [Pre-trained Models](#-pre-trained-models)
4. [Command Line Interface (CLI)](#-command-line-interface-cli)
5. [Advanced: Hosting Custom Models](#-advanced-hosting-custom-models)
6. [Resources & Tutorials](#-resources--tutorials)
7. [Releasing](#-releasing)

---

## 🚀 Installation

You can install `omniscore` directly via pip. The runtime dependencies include `sentencepiece` to ensure models like DeBERTa-v3 load cleanly.

```bash
pip install omniscore
```

If you want the latest GitHub version before a PyPI release, install it directly from the repository:

```bash
pip install "git+https://github.com/firojalam/omniscore_package.git"
```

### Source

[https://github.com/firojalam/omniscore_package](https://github.com/firojalam/omniscore_package)

*(Note: If installing from [source](https://github.com/firojalam/omniscore_package) for development, run `python3 -m pip install -e .` in the repository root.)*

### 30-Second Demo

List built-in known models:

```bash
omniscore --list-models --pretty
```

Run the documented example for a hosted model in one command:

```bash
omniscore \
  --model QCRI/OmniScore-deberta-v3 \
  --use-example \
  --pretty
```

---

## 💻 Quickstart: Python API

### Method 1: The `score()` Function
For one-off evaluations, you can use the high-level `score` function. It loads the model, processes the inputs, and returns the results.

```python
from omniscore import score

result = score(
    predictions=["A generated summary."],
    references=["A gold summary."],
    sources=["The source document."],
    tasks=["summarization"],
    model_name_or_path="your-org/omniscore-base",
)

print("Individual Scores:", result.to_list())
print("Mean Score:", result.mean())
```

### Method 2: The `OmniScorer` Class
If you are evaluating multiple batches or running a server, use `OmniScorer` to keep the model loaded in memory for much faster inference.

```python
from omniscore import OmniScorer

scorer = OmniScorer("your-org/omniscore-base")

result = scorer.score(
    predictions=["Candidate 1", "Candidate 2"],
    references=["Reference 1", "Reference 2"],
    sources=["Source 1", "Source 2"],
    tasks=["summarization", "summarization"],
)

print(result.to_list())
```

### Method 3: Score a Built-in Example
If you want to verify that a hosted model works end to end, use the built-in example helper.

```python
from omniscore import score_example

result = score_example("QCRI/OmniScore-deberta-v3", device="cpu", max_length=128)
print(result.to_json(indent=2))
```

---

## 🧠 Pre-trained Models

`omniscore` includes metadata for known hosted models and gracefully handles remote-code loading paths internally.

### Using `QCRI/OmniScore-deberta-v3`
Here is an example evaluating a headline using the legacy `score_predictor` checkpoint:

```python
from omniscore import OmniScorer, get_example

# Load built-in examples for the model
example = get_example("QCRI/OmniScore-deberta-v3")

# Initialize the scorer
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
result = scorer.score(
    predictions="Microsoft releases detailed model documentation.",
    sources="Full article text goes here.",
    tasks="headline_evaluation",
)
```

---

## ⌨️ Command Line Interface (CLI)

Evaluate text rapidly without writing Python scripts.

**Score a single example:**
```bash
omniscore \
  --model QCRI/OmniScore-deberta-v3 \
  --prediction "Microsoft releases detailed model documentation." \
  --source "Full article text goes here." \
  --task headline_evaluation \
  --pretty
```

**Score batches from files:**
```bash
omniscore \
  --model QCRI/OmniScore-deberta-v3 \
  --predictions-file predictions.txt \
  --references-file references.txt \
  --sources-file sources.txt \
  --tasks-file tasks.txt \
  --pretty
```

**Inspect model metadata and the built-in example:**
```bash
omniscore \
  --model QCRI/OmniScore-deberta-v3 \
  --show-model-info \
  --pretty
```

---

## 🛠 Advanced: Hosting Custom Models

`omniscore` is built around the standard Transformers save/load flow. You can easily adapt a backbone model and push it to the Hugging Face Hub.

### 1. Build and Save Locally
```python
from transformers import AutoTokenizer
from omniscore import OmniScoreModel

# Initialize an OmniScore model from a standard backbone
model = OmniScoreModel.from_backbone(
    "distilroberta-base",
    score_names=["quality", "faithfulness"],
    task_prefix="Task:",
    source_prefix="Document:",
    reference_prefix="Reference:",
    prediction_prefix="Summary:",
)

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

# Save standard checkpoints
model.save_pretrained("omniscore-checkpoint")
tokenizer.save_pretrained("omniscore-checkpoint")
```

### 2. Push to Hugging Face
Upload your folder to Hugging Face using standard `transformers` methods:

```python
model.push_to_hub("your-org/omniscore-base")
tokenizer.push_to_hub("your-org/omniscore-base")
```

Users can now load your model instantly using `OmniScorer("your-org/omniscore-base")`.

---

## 📚 Resources & Tutorials

Check out our clean, Colab-oriented example notebook located at:

- GitHub: [examples/omniscore_qcri_deberta_v3_colab.ipynb](https://github.com/firojalam/omniscore_package/blob/main/examples/omniscore_qcri_deberta_v3_colab.ipynb)
- Colab: [Open the notebook in Colab](https://colab.research.google.com/drive/1985As0_gN1mcRIt9Z_Q3jYRT2VBCzdo9)

This notebook walks you through:
- Verifying your GPU runtime.
- Loading the `QCRI/OmniScore-deberta-v3` model.
- Running documented examples via `OmniScorer`.

---

## 🚢 Releasing

`omniscore` now ships with a GitHub Actions release workflow that supports:

- automatic PyPI releases on tags like `v0.1.1`
- manual GitHub Actions releases to `testpypi` or `pypi`
- local manual uploads with `twine`

Setup and release instructions are documented in [docs/RELEASING.md](docs/RELEASING.md).
- Inspecting the returned score data structure.

### Supported Families
- Native `omniscore` checkpoints saved from `OmniScoreModel`.
- Legacy `score_predictor` checkpoints (e.g., `QCRI/OmniScore-deberta-v3`).
