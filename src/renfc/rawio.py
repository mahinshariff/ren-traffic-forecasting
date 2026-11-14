"""Streaming access to the raw per-router NetFlow Parquet files.

The raw files are large (the Dallas file alone holds ~150 M flow records), so
nothing in this module loads a whole router into memory. Instead,
:func:`iter_batches` yields pandas chunks with only the requested columns,
and callers accumulate whatever aggregate they need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def list_routers(raw_dir: str | Path) -> list[str]:
    """Router names available in ``raw_dir`` (``<name>.parquet`` files or dirs)."""
    raw_dir = Path(raw_dir)
    names = sorted(p.stem for p in raw_dir.glob("*.parquet"))
    return names


def router_path(raw_dir: str | Path, router: str) -> Path:
    path = Path(raw_dir) / f"{router}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No raw data for router '{router}' at {path}")
    return path


def available_columns(raw_dir: str | Path, router: str) -> list[str]:
    return ds.dataset(str(router_path(raw_dir, router)), format="parquet").schema.names


def iter_batches(
    raw_dir: str | Path,
    router: str,
    columns: list[str],
    batch_rows: int = 2_000_000,
) -> Iterator[pd.DataFrame]:
    """Yield pandas DataFrames of at most ``batch_rows`` rows with ``columns``.

    Columns that do not exist in the file are silently dropped from the
    request, so optional fields (e.g. ``in_bytes``) can be listed freely.
    """
    dataset = ds.dataset(str(router_path(raw_dir, router)), format="parquet")
    present = [c for c in columns if c in dataset.schema.names]
    scanner = dataset.scanner(columns=present, batch_size=batch_rows)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        yield batch.to_pandas()


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Convert a raw timestamp column to tz-naive ``datetime64[ns]``.

    Handles datetime columns, epoch seconds/milliseconds and strings; rows
    that cannot be parsed become ``NaT`` and are dropped by callers.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series)
    elif pd.api.types.is_numeric_dtype(series):
        values = series.astype("float64")
        # Heuristic: epoch milliseconds are > 1e11 for any date after 1973.
        unit = "ms" if np.nanmedian(values.values[: min(len(values), 10_000)]) > 1e11 else "s"
        out = pd.to_datetime(values, unit=unit, errors="coerce")
    else:
        out = pd.to_datetime(series, errors="coerce", format="mixed")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def ip_prefix(series: pd.Series, prefix_len: int = 16) -> pd.Series:
    """Map IPv4 addresses (dotted strings or 32-bit ints) to ``/prefix_len`` keys.

    The result is an integer key (the top ``prefix_len`` bits), which is cheap
    to group by and does not care how the anonymised addresses are encoded.
    """
    shift = 32 - prefix_len
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        values = series.fillna(0).to_numpy(dtype="int64")
        return pd.Series(np.right_shift(values, shift), index=series.index)
    # Strings: parse dotted quads without a Python loop per row.
    parts = series.astype("string").str.split(".", expand=True)
    if parts.shape[1] < 4:
        raise ValueError("Address column does not look like dotted IPv4 strings")
    octets = [pd.to_numeric(parts[i], errors="coerce").fillna(0).to_numpy(dtype="int64") for i in range(4)]
    as_int = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
    return pd.Series(np.right_shift(as_int, shift), index=series.index)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def table_num_rows(raw_dir: str | Path, router: str) -> int:
    """Row count from Parquet metadata (no data read)."""
    dataset = ds.dataset(str(router_path(raw_dir, router)), format="parquet")
    return sum(f.metadata.num_rows for f in dataset.get_fragments())


__all__ = [
    "list_routers",
    "router_path",
    "available_columns",
    "iter_batches",
    "parse_timestamps",
    "ip_prefix",
    "write_parquet",
    "read_parquet",
    "table_num_rows",
    "pa",
]
