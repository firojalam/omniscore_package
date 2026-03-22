"""CLI entrypoint for omniscore."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omniscore.scorer import OmniScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score text with a hosted OmniScore model.")
    parser.add_argument("--model", dest="model_name_or_path", default=None, help="HF repo id or local checkpoint path.")
    parser.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, mps.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer truncation length.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--prediction", help="Single prediction text to score.")
    parser.add_argument("--reference", default=None, help="Optional single reference text.")
    parser.add_argument("--source", default=None, help="Optional single source text.")
    parser.add_argument("--task", default=None, help="Optional task name such as headline_evaluation.")
    parser.add_argument("--predictions-file", type=Path, default=None, help="File with one prediction per line.")
    parser.add_argument("--references-file", type=Path, default=None, help="File with one reference per line.")
    parser.add_argument("--sources-file", type=Path, default=None, help="File with one source per line.")
    parser.add_argument("--tasks-file", type=Path, default=None, help="File with one task name per line.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    predictions, references, sources, tasks = _resolve_inputs(args, parser)
    scorer = OmniScorer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    result = scorer.score(
        predictions,
        references=references,
        sources=sources,
        tasks=tasks,
    )

    payload = {
        "score_names": list(result.score_names),
        "scores": result.to_list(),
        "mean": result.mean(),
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False))


def _resolve_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.prediction is not None:
        if args.predictions_file is not None:
            parser.error("Use either --prediction or --predictions-file, not both.")
        if args.tasks_file is not None:
            parser.error("Use either --task or --tasks-file, not both.")
        return args.prediction, args.reference, args.source, args.task

    if args.predictions_file is None:
        parser.error("Pass --prediction for a single example or --predictions-file for batch scoring.")

    predictions = _read_lines(args.predictions_file)
    references = _read_lines(args.references_file) if args.references_file else None
    sources = _read_lines(args.sources_file) if args.sources_file else None
    tasks = _read_lines(args.tasks_file) if args.tasks_file else None
    return predictions, references, sources, tasks


def _read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    main()
