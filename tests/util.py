import os
import tempfile

_COUNTER = [0]


def write_temp(content: str, suffix: str = ".csv") -> str:
    _COUNTER[0] += 1
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"pca_test_{_COUNTER[0]}_")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return path
