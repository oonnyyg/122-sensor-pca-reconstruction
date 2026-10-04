"""Small pure-Python linear algebra helpers (no third-party numeric libs)."""

from __future__ import annotations

import math


def column_means(rows: list[list[float]]) -> list[float]:
    n = len(rows)
    width = len(rows[0])
    return [sum(row[j] for row in rows) / n for j in range(width)]


def column_stdevs(rows: list[list[float]], means: list[float]) -> list[float]:
    """Sample standard deviation with denominator (n - 1)."""
    n = len(rows)
    if n < 2:
        raise ValueError("sample standard deviation needs at least two rows")
    width = len(rows[0])
    stdevs = []
    for j in range(width):
        ss = sum((row[j] - means[j]) ** 2 for row in rows)
        stdevs.append(math.sqrt(ss / (n - 1)))
    return stdevs


def standardize(
    rows: list[list[float]], means: list[float], stdevs: list[float]
) -> list[list[float]]:
    """Z-score rows; constant columns (stdev == 0) map to 0.0."""
    out = []
    for row in rows:
        out.append(
            [
                (value - means[j]) / stdevs[j] if stdevs[j] > 0.0 else 0.0
                for j, value in enumerate(row)
            ]
        )
    return out


def covariance_matrix(zrows: list[list[float]]) -> list[list[float]]:
    """Sample covariance of standardized rows, denominator (n - 1)."""
    n = len(zrows)
    width = len(zrows[0])
    cov = [[0.0] * width for _ in range(width)]
    for row in zrows:
        for i in range(width):
            ri = row[i]
            if ri == 0.0:
                continue
            for j in range(i, width):
                cov[i][j] += ri * row[j]
    for i in range(width):
        for j in range(i, width):
            cov[i][j] /= n - 1
            cov[j][i] = cov[i][j]
    return cov


def jacobi_eigh(
    matrix: list[list[float]], *, max_sweeps: int = 100, tol: float = 1e-12
) -> tuple[list[float], list[list[float]]]:
    """Eigen-decomposition of a real symmetric matrix via cyclic Jacobi rotations.

    Returns (eigenvalues, eigenvectors) with eigenvectors stored as columns,
    i.e. ``vectors[row][k]`` is component ``row`` of eigenvector ``k``.
    Eigenvalues are sorted in descending order; equal eigenvalues keep the
    order produced by the deterministic rotation sequence.
    """
    size = len(matrix)
    a = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]

    def off_diagonal_norm() -> float:
        return math.sqrt(
            sum(a[i][j] ** 2 for i in range(size) for j in range(size) if i != j)
        )

    for _ in range(max_sweeps):
        if off_diagonal_norm() <= tol:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if abs(a[p][q]) <= tol * 1e-3:
                    continue
                app, aqq, apq = a[p][p], a[q][q], a[p][q]
                theta = 0.5 * (aqq - app) / apq
                t = math.copysign(1.0, theta) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0)
                )
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(size):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(size):
                    vkp, vkq = vectors[k][p], vectors[k][q]
                    vectors[k][p] = c * vkp - s * vkq
                    vectors[k][q] = s * vkp + c * vkq

    eigenvalues = [a[i][i] for i in range(size)]
    order = sorted(range(size), key=lambda k: -eigenvalues[k])
    sorted_values = [eigenvalues[k] for k in order]
    sorted_vectors = [[vectors[i][k] for k in order] for i in range(size)]
    return sorted_values, sorted_vectors


def canonicalize_sign(vector: list[float]) -> list[float]:
    """Flip the vector so its largest-|component| entry is positive.

    Ties in absolute value are broken by original column order (the first
    index with maximal magnitude wins).
    """
    best = max(range(len(vector)), key=lambda i: abs(vector[i]))
    if vector[best] < 0.0:
        return [-v for v in vector]
    return list(vector)
