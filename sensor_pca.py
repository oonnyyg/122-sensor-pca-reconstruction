"""PCA compression / reconstruction tool for multi-sensor CSV data.

Pure standard library implementation. See README.md for numerical notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys


class PCAError(Exception):
    """Raised for invalid input data or model misuse."""


def load_csv(path):
    """Read a sensor CSV.

    First column is the record id; remaining columns are named numeric
    features. Duplicate column names and non-finite values are rejected.
    Returns (ids, columns, rows) where rows is a list of float lists.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise PCAError(f"{path}: empty CSV")
        if len(header) < 2:
            raise PCAError(f"{path}: need an id column plus at least one value column")
        columns = header[1:]
        if len(set(columns)) != len(columns):
            raise PCAError(f"{path}: duplicate column names: {columns}")
        ids = []
        rows = []
        for lineno, record in enumerate(reader, start=2):
            if not record:
                continue
            if len(record) != len(header):
                raise PCAError(f"{path}:{lineno}: expected {len(header)} fields, got {len(record)}")
            values = []
            for name, text in zip(columns, record[1:]):
                try:
                    value = float(text)
                except ValueError:
                    raise PCAError(f"{path}:{lineno}: column {name!r} is not numeric: {text!r}")
                if not math.isfinite(value):
                    raise PCAError(f"{path}:{lineno}: column {name!r} is not finite: {text!r}")
                values.append(value)
            ids.append(record[0])
            rows.append(values)
    if not rows:
        raise PCAError(f"{path}: no data rows")
    return ids, columns, rows


def _column_stats(rows):
    """Sample mean and std (denominator n-1) per column."""
    n = len(rows)
    means = [sum(col) / n for col in zip(*rows)]
    scales = []
    for j, mean in enumerate(means):
        var = sum((row[j] - mean) ** 2 for row in rows) / (n - 1)
        scales.append(math.sqrt(var))
    return means, scales


def _covariance(zrows):
    """Sample covariance matrix of already-standardized rows."""
    n = len(zrows)
    p = len(zrows[0])
    cov = [[0.0] * p for _ in range(p)]
    for row in zrows:
        for i in range(p):
            ri = row[i]
            for j in range(i, p):
                cov[i][j] += ri * row[j]
    for i in range(p):
        for j in range(i, p):
            cov[i][j] /= n - 1
            cov[j][i] = cov[i][j]
    return cov


def _jacobi_eigh(matrix, sweeps=100, tol=1e-15):
    """Eigenvalues and unit eigenvectors of a symmetric matrix.

    Classic cyclic Jacobi rotations. Returns (eigenvalues, vectors) where
    vectors[k] is the eigenvector for eigenvalues[k], unsorted.
    """
    p = len(matrix)
    a = [row[:] for row in matrix]
    vecs = [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    for _ in range(sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(p) for j in range(p) if i != j))
        if off <= tol * max(1.0, math.sqrt(sum(a[i][i] ** 2 for i in range(p)))):
            break
        for i in range(p - 1):
            for j in range(i + 1, p):
                if a[i][j] == 0.0:
                    continue
                theta = (a[j][j] - a[i][i]) / (2.0 * a[i][j])
                t = math.copysign(1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0)), theta)
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(p):
                    aik, ajk = a[i][k], a[j][k]
                    a[i][k] = c * aik - s * ajk
                    a[j][k] = s * aik + c * ajk
                for k in range(p):
                    aki, akj = a[k][i], a[k][j]
                    a[k][i] = c * aki - s * akj
                    a[k][j] = s * aki + c * akj
                for k in range(p):
                    vki, vkj = vecs[k][i], vecs[k][j]
                    vecs[k][i] = c * vki - s * vkj
                    vecs[k][j] = s * vki + c * vkj
    eigenvalues = [a[i][i] for i in range(p)]
    vectors = [[vecs[i][j] for i in range(p)] for j in range(p)]
    return eigenvalues, vectors


def fit(rows, columns, variance_ratio):
    """Fit a PCA model and return it as a JSON-serializable dict."""
    if not (0.0 < variance_ratio <= 1.0):
        raise PCAError("variance ratio must be > 0 and <= 1")
    if len(rows) < 2:
        raise PCAError("fitting requires at least two rows")
    means, scales = _column_stats(rows)
    active = [j for j, s in enumerate(scales) if s > 0.0]
    if not active:
        raise PCAError("all columns are constant; no principal components exist")
    zrows = [[(row[j] - means[j]) / scales[j] for j in active] for row in rows]
    cov = _covariance(zrows)
    eigenvalues, vectors = _jacobi_eigh(cov)
    # Deterministic order: eigenvalue descending; ties keep the original
    # column order (stable sort), so repeated eigenvalues are reproducible.
    order = sorted(range(len(eigenvalues)), key=lambda k: -eigenvalues[k])
    total = sum(max(v, 0.0) for v in eigenvalues)
    kept = []
    cumulative = 0.0
    for k in order:
        value = max(eigenvalues[k], 0.0)
        vec = vectors[k]
        # Sign convention: the element with the largest absolute value is
        # positive; ties resolve to the earliest original column.
        pivot = max(range(len(vec)), key=lambda i: abs(vec[i]))
        if vec[pivot] < 0.0:
            vec = [-x for x in vec]
        kept.append({"eigenvalue": value, "vector": vec})
        cumulative += value
        if total > 0.0 and cumulative / total >= variance_ratio:
            break
    return {
        "format": "sensor-pca-model/1",
        "columns": list(columns),
        "means": means,
        "scales": scales,
        "active_columns": [columns[j] for j in active],
        "variance_ratio": variance_ratio,
        "total_variance": total,
        "components": [
            {
                "eigenvalue": item["eigenvalue"],
                "explained_ratio": item["eigenvalue"] / total if total > 0.0 else 0.0,
                "vector": item["vector"],
            }
            for item in kept
        ],
    }


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
        fh.write("\n")


def load_model(path):
    with open(path, encoding="utf-8") as fh:
        model = json.load(fh)
    if model.get("format") != "sensor-pca-model/1":
        raise PCAError(f"{path}: not a sensor-pca-model/1 file")
    for key in ("columns", "means", "scales", "active_columns", "components"):
        if key not in model:
            raise PCAError(f"{path}: model missing key {key!r}")
    return model


def apply_model(model, ids, columns, rows):
    """Project and reconstruct rows; return per-record result dicts."""
    model_columns = model["columns"]
    if set(columns) != set(model_columns) or len(columns) != len(model_columns):
        missing = [c for c in model_columns if c not in columns]
        extra = [c for c in columns if c not in model_columns]
        raise PCAError(f"column mismatch: missing={missing} extra={extra}")
    index = {name: columns.index(name) for name in model_columns}
    means = model["means"]
    scales = model["scales"]
    active_idx = [model_columns.index(c) for c in model["active_columns"]]
    components = model["components"]
    results = []
    for rec_id, row in zip(ids, rows):
        aligned = [row[index[name]] for name in model_columns]
        z = [(aligned[j] - means[j]) / scales[j] for j in active_idx]
        scores = [sum(zi * vi for zi, vi in zip(z, comp["vector"])) for comp in components]
        z_hat = [0.0] * len(active_idx)
        for score, comp in zip(scores, components):
            for k, vi in enumerate(comp["vector"]):
                z_hat[k] += score * vi
        recon = list(means)
        for k, j in enumerate(active_idx):
            recon[j] = z_hat[k] * scales[j] + means[j]
        # Constant columns keep their fitted mean; deviations there still
        # contribute to the Euclidean reconstruction error.
        error = math.sqrt(sum((a - r) ** 2 for a, r in zip(aligned, recon)))
        results.append({"id": rec_id, "scores": scores, "reconstruction": recon, "error": error})
    return results


def write_results(path, model, results):
    score_names = [f"score_pc{k + 1}" for k in range(len(model["components"]))]
    recon_names = [f"recon_{c}" for c in model["columns"]]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", *score_names, *recon_names, "reconstruction_error"])
        for res in results:
            writer.writerow(
                [res["id"]]
                + [repr(v) for v in res["scores"]]
                + [repr(v) for v in res["reconstruction"]]
                + [repr(res["error"])]
            )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sensor_pca", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_fit = sub.add_parser("fit", help="fit a PCA model from a CSV")
    p_fit.add_argument("csv")
    p_fit.add_argument("model", help="output JSON model path")
    p_fit.add_argument("--variance-ratio", type=float, default=0.95,
                       help="cumulative explained variance target, in (0, 1] (default 0.95)")
    p_apply = sub.add_parser("apply", help="project and reconstruct a CSV with a saved model")
    p_apply.add_argument("csv")
    p_apply.add_argument("model")
    p_apply.add_argument("output", help="output CSV path")
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            _, columns, rows = load_csv(args.csv)
            model = fit(rows, columns, args.variance_ratio)
            save_model(model, args.model)
            kept = len(model["components"])
            total = len(model["active_columns"])
            print(f"fitted {kept} of {total} components "
                  f"({len(model['columns']) - total} constant columns excluded)")
        else:
            model = load_model(args.model)
            ids, columns, rows = load_csv(args.csv)
            results = apply_model(model, ids, columns, rows)
            write_results(args.output, model, results)
            print(f"wrote {len(results)} records to {args.output}")
    except PCAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
