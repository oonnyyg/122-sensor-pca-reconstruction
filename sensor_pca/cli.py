"""Command line interface: fit a PCA model or apply a saved one."""

from __future__ import annotations

import argparse
import sys

from .data import DataError, read_csv, write_csv
from .model import PCAModel


def _cmd_fit(args: argparse.Namespace) -> int:
    ids, columns, rows = read_csv(args.input)
    if len(rows) < 2:
        raise DataError(f"{args.input}: fitting requires at least two data rows")
    model = PCAModel.fit(columns, rows, args.variance_ratio)
    model.save(args.model)
    kept = len(model.components)
    total_ratio = sum(model.explained_ratio)
    print(
        f"fitted {len(rows)} records x {len(columns)} columns; "
        f"kept {kept} component(s) covering {total_ratio:.6f} of variance; "
        f"model saved to {args.model}"
    )
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    model = PCAModel.load(args.model)
    ids, columns, rows = read_csv(args.input)
    scores, recon, errors = model.apply(columns, rows)

    score_header = ["id"] + [f"pc{k + 1}" for k in range(len(model.components))]
    write_csv(
        args.scores,
        score_header,
        [
            [record_id] + [f"{value:.17g}" for value in row]
            for record_id, row in zip(ids, scores)
        ],
    )
    recon_header = ["id"] + model.columns + ["reconstruction_error"]
    write_csv(
        args.reconstruction,
        recon_header,
        [
            [record_id]
            + [f"{value:.17g}" for value in row]
            + [f"{error:.17g}"]
            for record_id, row, error in zip(ids, recon, errors)
        ],
    )
    print(
        f"applied model to {len(ids)} records; "
        f"scores -> {args.scores}; reconstruction -> {args.reconstruction}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sensor-pca",
        description="Pure-stdlib PCA compression and reconstruction for sensor CSV data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit a PCA model and save it as JSON")
    fit.add_argument("--input", required=True, help="training CSV path")
    fit.add_argument("--model", required=True, help="output model JSON path")
    fit.add_argument(
        "--variance-ratio",
        type=float,
        required=True,
        help="cumulative explained variance target, in (0, 1]",
    )
    fit.set_defaults(func=_cmd_fit)

    apply_cmd = sub.add_parser(
        "apply", help="project and reconstruct with a saved model"
    )
    apply_cmd.add_argument("--model", required=True, help="model JSON path")
    apply_cmd.add_argument("--input", required=True, help="CSV path to transform")
    apply_cmd.add_argument(
        "--scores", required=True, help="output CSV for projection scores"
    )
    apply_cmd.add_argument(
        "--reconstruction",
        required=True,
        help="output CSV for original-unit reconstruction and per-record error",
    )
    apply_cmd.set_defaults(func=_cmd_apply)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
