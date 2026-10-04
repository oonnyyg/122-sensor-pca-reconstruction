"""CSV loading and validation for sensor records."""

from __future__ import annotations

import csv
import math


class DataError(ValueError):
    """Raised when input data fails validation."""


def read_csv(path: str) -> tuple[list[str], list[str], list[list[float]]]:
    """Read a sensor CSV file.

    The first column is a unique record id; the remaining columns are named
    numeric features. Returns (ids, column_names, rows).
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise DataError(f"{path}: empty CSV file") from None
        header = [cell.strip() for cell in header]
        if len(header) < 2:
            raise DataError(f"{path}: need an id column plus at least one value column")
        names = header[1:]
        if any(not name for name in names):
            raise DataError(f"{path}: column names must be non-empty")
        if len(set(names)) != len(names):
            raise DataError(f"{path}: duplicate column names are not allowed")

        ids: list[str] = []
        rows: list[list[float]] = []
        seen_ids: set[str] = set()
        for line_no, record in enumerate(reader, start=2):
            if not record or all(not cell.strip() for cell in record):
                continue
            if len(record) != len(header):
                raise DataError(
                    f"{path}: line {line_no} has {len(record)} fields, "
                    f"expected {len(header)}"
                )
            record_id = record[0].strip()
            if not record_id:
                raise DataError(f"{path}: line {line_no} has an empty record id")
            if record_id in seen_ids:
                raise DataError(f"{path}: duplicate record id {record_id!r}")
            seen_ids.add(record_id)
            values: list[float] = []
            for name, cell in zip(names, record[1:]):
                text = cell.strip()
                try:
                    value = float(text)
                except ValueError:
                    raise DataError(
                        f"{path}: line {line_no} column {name!r} is not numeric: {text!r}"
                    ) from None
                if not math.isfinite(value):
                    raise DataError(
                        f"{path}: line {line_no} column {name!r} is not finite: {text!r}"
                    )
                values.append(value)
            ids.append(record_id)
            rows.append(values)
    if not rows:
        raise DataError(f"{path}: no data rows found")
    return ids, names, rows


def write_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
