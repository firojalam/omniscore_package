"""CLI entrypoint for omniscore."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omniscore import __version__
from omniscore.examples import get_example, get_known_model, iter_known_models
from omniscore.scorer import OmniScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score text with a hosted OmniScore model.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("--model", dest="model_name_or_path", default=None, help="HF repo id or local checkpoint path.")
    parser.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, mps.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer truncation length.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--revision", default=None, help="Optional model revision, branch, or commit id.")
    parser.add_argument("--list-models", action="store_true", help="List built-in known models and exit.")
    parser.add_argument("--show-model-info", action="store_true", help="Show metadata for --model and exit.")
    parser.add_argument("--use-example", action="store_true", help="Score the built-in example for --model.")
    parser.add_argument("--prediction", help="Single prediction text to score.")
    parser.add_argument("--reference", default=None, help="Optional single reference text.")
    parser.add_argument("--source", default=None, help="Optional single source text.")
    parser.add_argument("--task", default=None, help="Optional task name such as headline_evaluation.")
    parser.add_argument("--predictions-file", type=Path, default=None, help="File with one prediction per line.")
    parser.add_argument("--references-file", type=Path, default=None, help="File with one reference per line.")
    parser.add_argument("--sources-file", type=Path, default=None, help="File with one source per line.")
    parser.add_argument("--tasks-file", type=Path, default=None, help="File with one task name per line.")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional path to write JSON output.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        _emit_payload({"models": [model.to_dict() for model in iter_known_models()]}, args)
        return

    if args.show_model_info:
        if args.model_name_or_path is None:
            parser.error("--show-model-info requires --model.")
        known_model = get_known_model(args.model_name_or_path)
        if known_model is None:
            parser.error(f"No built-in model metadata is registered for {args.model_name_or_path!r}.")
        _emit_payload({"model": known_model.to_dict()}, args)
        return

    predictions, references, sources, tasks, example = _resolve_inputs(args, parser)
    scorer = OmniScorer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )

    if example is not None:
        result = scorer.score_example(args.model_name_or_path)
        payload: dict[str, Any] = {
            "model": scorer.model_name_or_path,
            "example": example.to_dict(),
            **result.to_dict(),
        }
    else:
        result = scorer.score(
            predictions,
            references=references,
            sources=sources,
            tasks=tasks,
        )
        payload = {
            "model": scorer.model_name_or_path,
            **result.to_dict(),
        }

    known_model = get_known_model(scorer.model_name_or_path)
    if known_model is not None:
        payload["known_model"] = {
            "repo_id": known_model.repo_id,
            "family": known_model.family,
            "tasks": list(known_model.tasks),
            "model_card_url": known_model.model_card_url,
        }

    _emit_payload(payload, args)


def _resolve_inputs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[str | list[str] | None, str | list[str] | None, str | list[str] | None, str | list[str] | None, Any | None]:
    if args.use_example:
        if args.model_name_or_path is None:
            parser.error("--use-example requires --model.")
        _ensure_no_manual_inputs(args, parser)
        example = get_example(args.model_name_or_path)
        if example is None:
            parser.error(f"No built-in example is registered for {args.model_name_or_path!r}.")
        return None, None, None, None, example

    if args.prediction is not None:
        if args.predictions_file is not None:
            parser.error("Use either --prediction or --predictions-file, not both.")
        if args.tasks_file is not None:
            parser.error("Use either --task or --tasks-file, not both.")
        return args.prediction, args.reference, args.source, args.task, None

    if args.predictions_file is None:
        parser.error(
            "Pass --prediction for a single example, --predictions-file for batch scoring, "
            "or --use-example for a built-in demo."
        )

    predictions = _read_lines(args.predictions_file)
    references = _read_lines(args.references_file) if args.references_file else None
    sources = _read_lines(args.sources_file) if args.sources_file else None
    tasks = _read_lines(args.tasks_file) if args.tasks_file else None
    return predictions, references, sources, tasks, None


def _ensure_no_manual_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    conflicting_flags = (
        args.prediction,
        args.reference,
        args.source,
        args.task,
        args.predictions_file,
        args.references_file,
        args.sources_file,
        args.tasks_file,
    )
    if any(value is not None for value in conflicting_flags):
        parser.error("--use-example cannot be combined with manual input flags.")


def _emit_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    rendered = json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)
    print(rendered)
    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(f"{rendered}\n", encoding="utf-8")


def _read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    main()
