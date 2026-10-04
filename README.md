# sensor-pca-reconstruction

Pure-standard-library PCA compression and reconstruction for multi-sensor
CSV data. No third-party numeric packages.

## CSV format

- First column: unique record id.
- Remaining columns: named numeric features. Duplicate column names and
  non-finite values (`nan`, `inf`, text) are rejected.

## Usage

```sh
python sensor_pca.py fit train.csv model.json --variance-ratio 0.95
python sensor_pca.py apply new_data.csv model.json results.csv
```

`fit` requires at least two rows. It standardizes each column with the
sample mean and the sample standard deviation (denominator `n - 1`),
builds the sample covariance matrix of the standardized data, and keeps
the smallest set of principal components whose cumulative explained
variance ratio reaches `--variance-ratio` (must be in `(0, 1]`).

`apply` aligns columns by name (order may differ; missing or extra
columns are rejected), never re-fits, and writes a CSV with one row per
record: `score_pc1..score_pcK`, `recon_<column>` in original units, and
`reconstruction_error` (Euclidean distance in original units). Constant
columns are reconstructed as their fitted mean, so a changed constant
column contributes to the error.

## Model file (JSON)

Stores column names, per-column means and scales, the active
(non-constant) column list, and each kept component's eigenvalue,
explained variance ratio, and unit direction vector.

## Numerical notes

- Constant columns (zero sample std) are excluded from the covariance
  solve but their means are kept in the model. If *all* columns are
  constant, fitting is rejected: there is no information to compress.
- Eigendecomposition uses the classic cyclic Jacobi rotation algorithm
  for symmetric matrices (sweep until the off-diagonal norm falls below
  `1e-15` relative to the diagonal norm, max 100 sweeps). Accuracy is
  typically near machine epsilon for well-scaled problems, but all
  arithmetic is pure-Python double precision and slower than BLAS-based
  libraries; very large or ill-conditioned matrices will lose precision.
- Components are sorted by eigenvalue descending. Ties (repeated
  eigenvalues) are resolved by a stable sort that preserves the original
  column order, so results are deterministic run to run; note that any
  orthonormal basis of a degenerate eigenspace is equally valid.
- Sign convention: each eigenvector's largest-magnitude element is made
  positive; ties pick the earliest original column. This makes saved
  models reproducible.
- CSV output writes floats via `repr()`, which round-trips Python
  doubles exactly.

## Tests

```sh
python -m unittest test_sensor_pca -v
```

Covers CSV validation, correlated columns, constant columns, component
ordering/orthonormality, model save/reload, column reordering, and
reconstruction error accounting.
