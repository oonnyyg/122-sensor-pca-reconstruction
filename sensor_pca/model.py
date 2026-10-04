"""PCA model: fit, serialize, project and reconstruct."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .data import DataError
from .linalg import (
    canonicalize_sign,
    column_means,
    column_stdevs,
    covariance_matrix,
    jacobi_eigh,
    standardize,
)

MODEL_FORMAT = "sensor-pca-model"
MODEL_VERSION = 1


@dataclass
class PCAModel:
    columns: list[str]
    means: list[float]
    scales: list[float]
    constant: list[bool]
    components: list[list[float]]  # each entry is one unit eigenvector
    explained_ratio: list[float]  # variance share per retained component
    variance_ratio: float  # configured cumulative target

    @classmethod
    def fit(
        cls, columns: list[str], rows: list[list[float]], variance_ratio: float
    ) -> "PCAModel":
        if not 0.0 < variance_ratio <= 1.0:
            raise DataError(
                "variance ratio must be greater than 0 and at most 1, "
                f"got {variance_ratio!r}"
            )
        if len(rows) < 2:
            raise DataError("fitting requires at least two data rows")
        means = column_means(rows)
        scales = column_stdevs(rows, means)
        constant = [s == 0.0 for s in scales]
        if all(constant):
            raise DataError(
                "all columns are constant; no principal components can be derived"
            )
        zrows = standardize(rows, means, scales)
        cov = covariance_matrix(zrows)
        eigenvalues, eigvecs = jacobi_eigh(cov)
        total = sum(v for v in eigenvalues if v > 0.0)
        if total <= 0.0:
            raise DataError("covariance matrix carries no variance")

        cumulative = 0.0
        keep = 0
        for value in eigenvalues:
            if value <= 0.0:
                break
            cumulative += value
            keep += 1
            if cumulative / total >= variance_ratio:
                break

        components = []
        explained = []
        for k in range(keep):
            vector = canonicalize_sign([eigvecs[i][k] for i in range(len(columns))])
            components.append(vector)
            explained.append(eigenvalues[k] / total)
        return cls(
            columns=list(columns),
            means=means,
            scales=scales,
            constant=constant,
            components=components,
            explained_ratio=explained,
            variance_ratio=variance_ratio,
        )

    # ---- serialization -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": MODEL_FORMAT,
            "version": MODEL_VERSION,
            "variance_ratio": self.variance_ratio,
            "columns": self.columns,
            "means": self.means,
            "scales": self.scales,
            "constant": self.constant,
            "components": self.components,
            "explained_ratio": self.explained_ratio,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PCAModel":
        if payload.get("format") != MODEL_FORMAT:
            raise DataError("not a sensor-pca model file")
        if payload.get("version") != MODEL_VERSION:
            raise DataError(f"unsupported model version {payload.get('version')!r}")
        try:
            model = cls(
                columns=[str(c) for c in payload["columns"]],
                means=[float(v) for v in payload["means"]],
                scales=[float(v) for v in payload["scales"]],
                constant=[bool(v) for v in payload["constant"]],
                components=[
                    [float(x) for x in vec] for vec in payload["components"]
                ],
                explained_ratio=[float(v) for v in payload["explained_ratio"]],
                variance_ratio=float(payload["variance_ratio"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"malformed model file: {exc}") from None
        width = len(model.columns)
        if not (
            len(model.means)
            == len(model.scales)
            == len(model.constant)
            == width
        ):
            raise DataError("model file has inconsistent column widths")
        if len(model.components) != len(model.explained_ratio):
            raise DataError("model components and explained ratios differ in length")
        for vec in model.components:
            if len(vec) != width:
                raise DataError("model component width does not match columns")
        return model

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str) -> "PCAModel":
        with open(path, encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}: invalid JSON: {exc}") from None
        if not isinstance(payload, dict):
            raise DataError(f"{path}: model must be a JSON object")
        return cls.from_dict(payload)

    # ---- application ---------------------------------------------------

    def align_rows(
        self, columns: list[str], rows: list[list[float]]
    ) -> list[list[float]]:
        """Reorder incoming rows to model column order by name."""
        incoming = set(columns)
        expected = set(self.columns)
        missing = [c for c in self.columns if c not in incoming]
        extra = [c for c in columns if c not in expected]
        if missing:
            raise DataError(f"missing columns: {', '.join(missing)}")
        if extra:
            raise DataError(f"unexpected columns: {', '.join(extra)}")
        index = [columns.index(name) for name in self.columns]
        return [[row[j] for j in index] for row in rows]

    def project_row(self, row: list[float]) -> list[float]:
        z = [
            (value - self.means[j]) / self.scales[j]
            if not self.constant[j]
            else 0.0
            for j, value in enumerate(row)
        ]
        return [
            sum(z[j] * component[j] for j in range(len(z)))
            for component in self.components
        ]

    def reconstruct_row(self, scores: list[float]) -> list[float]:
        out = []
        for j in range(len(self.columns)):
            if self.constant[j]:
                out.append(self.means[j])
                continue
            z = sum(
                scores[k] * self.components[k][j]
                for k in range(len(self.components))
            )
            out.append(z * self.scales[j] + self.means[j])
        return out

    def apply(
        self, columns: list[str], rows: list[list[float]]
    ) -> tuple[list[list[float]], list[list[float]], list[float]]:
        """Return (scores, reconstruction, euclidean error) per record."""
        aligned = self.align_rows(columns, rows)
        all_scores: list[list[float]] = []
        all_recon: list[list[float]] = []
        all_errors: list[float] = []
        for row in aligned:
            scores = self.project_row(row)
            recon = self.reconstruct_row(scores)
            error = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(row, recon))
            )
            all_scores.append(scores)
            all_recon.append(recon)
            all_errors.append(error)
        return all_scores, all_recon, all_errors
