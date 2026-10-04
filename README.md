# sensor-pca-reconstruction

Pure-Python (standard library only) PCA tool for compressing multi-channel
sensor CSV data and measuring reconstruction deviation. No third-party
numeric libraries are used; eigen-decomposition is done with cyclic Jacobi
rotations on the sample covariance matrix.

## Input format

- CSV with a header row; the first column is a unique record id, the rest
  are named numeric columns.
- Duplicate column names, duplicate ids, non-numeric and non-finite
  (`nan`/`inf`) values are rejected.
- Fitting requires at least two data rows.

## Usage

Fit a model and save it as JSON:

```
python -m sensor_pca fit --input train.csv --model model.json --variance-ratio 0.95
```

Apply a saved model to new data (columns are aligned by name, so the order
may differ; missing or extra columns are rejected):

```
python -m sensor_pca apply --model model.json --input new.csv \
    --scores scores.csv --reconstruction recon.csv
```

Outputs:

- `scores.csv`: `id,pc1,...,pcK` projection scores per record.
- `recon.csv`: `id,<original columns...>,reconstruction_error` — the
  reconstruction in original units plus the per-record Euclidean error in
  original units. A changed constant column contributes to this error.

## Method

- Standardization uses the sample mean and the sample standard deviation
  with denominator `n - 1`.
- Constant columns (zero variance) are excluded from the covariance solve;
  the model keeps their mean and restores that value on reconstruction.
  If every column is constant, fitting is rejected.
- Eigenvalues/eigenvectors come from the covariance matrix of the
  standardized data, sorted by descending eigenvalue. Each eigenvector is
  re-signed so its largest-|component| entry is positive; ties are broken
  by original column order (first maximal index wins). Repeated eigenvalues
  are handled deterministically: the Jacobi sweep order is fixed and the
  descending sort is stable, so equal eigenvalues always keep the same
  relative order for identical input.
- The number of retained components is the smallest count whose cumulative
  explained variance ratio reaches `--variance-ratio` (must be in `(0, 1]`).
- The JSON model stores column names, means, scales, constant-column flags,
  retained unit directions and their explained-variance ratios. Applying a
  model never re-fits.

## Numerical precision

All computation uses IEEE-754 double precision (`float`). Jacobi iteration
stops when the off-diagonal norm drops below `1e-12` or after 100 sweeps,
so eigenvalues and scores are typically accurate to about 1e-10 relative
error; near-degenerate (repeated) eigenvalues may define any orthonormal
basis of their shared subspace, but the choice is deterministic for a
given input. CSV outputs are written with `%.17g` formatting, which
round-trips doubles exactly.

## Tests

```
python -m unittest discover -s tests
```
