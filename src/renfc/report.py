"""Stage reports.

Each stage produces one Markdown file under ``<out_dir>/reports``.  The file is
deliberately short (a few hundred lines at most) and self-contained, so the
whole state of a run can be reviewed by reading the reports alone.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, out_dir


class Report:
    """Accumulates Markdown sections and writes them at the end of a stage."""

    def __init__(self, cfg: Config, stage: str, title: str):
        self.cfg = cfg
        self.stage = stage
        self.title = title
        self.lines: list[str] = [f"# {title}", "", f"_Generated {dt.datetime.now():%Y-%m-%d %H:%M}_", ""]
        self.meta: dict[str, Any] = {}

    def section(self, heading: str) -> None:
        self.lines += ["", f"## {heading}", ""]

    def text(self, *paragraphs: str) -> None:
        for p in paragraphs:
            self.lines += [p, ""]

    def bullet(self, item: str) -> None:
        self.lines.append(f"- {item}")

    def kv(self, key: str, value: Any) -> None:
        self.lines.append(f"- **{key}:** {value}")
        self.meta[key] = value if isinstance(value, (int, float, str, bool, list, dict)) else str(value)

    def table(self, df: pd.DataFrame, max_rows: int = 60, floatfmt: str = ".4g") -> None:
        """Append a DataFrame as a Markdown table (truncated to ``max_rows``)."""
        if df is None or len(df) == 0:
            self.lines += ["_(empty)_", ""]
            return
        shown = df.head(max_rows).copy()
        for col in shown.columns:
            if pd.api.types.is_float_dtype(shown[col]):
                shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
        header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
        sep = "|" + "|".join(["---"] * len(shown.columns)) + "|"
        rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in shown.itertuples(index=False)]
        self.lines += [header, sep, *rows]
        if len(df) > max_rows:
            self.lines.append(f"\n_({len(df) - max_rows} more rows not shown)_")
        self.lines.append("")

    def write(self) -> Path:
        path = out_dir(self.cfg, "reports") / f"{self.stage}.md"
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        return path
