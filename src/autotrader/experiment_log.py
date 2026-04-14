"""Experiment log: TSV-based results tracking.

Direct port of autoresearch-mlx's results.tsv pattern. Append-only,
tab-separated, human-readable, git-trackable.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from .metrics import ExperimentResult


TSV_COLUMNS = [
    "experiment_id",
    "thesis_id",
    "spec_id",
    "composite_score",
    "sharpe_ratio",
    "max_drawdown",
    "hit_rate",
    "cumulative_return",
    "num_trades",
    "backtest_duration_seconds",
    "status",
    "description",
]


class ExperimentLog:
    """Append-only TSV experiment log, mirroring autoresearch's results.tsv."""

    def __init__(self, path: str | Path = "data/autotrader/experiment_results.tsv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_header()

    def _write_header(self) -> None:
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(TSV_COLUMNS)

    def append(self, result: ExperimentResult) -> None:
        """Append a single experiment result."""
        row = [
            result.experiment_id,
            result.thesis_id,
            result.spec_id,
            f"{result.composite_score:.6f}",
            f"{result.sharpe_ratio:.4f}",
            f"{result.max_drawdown:.4f}",
            f"{result.hit_rate:.4f}",
            f"{result.cumulative_return:.6f}",
            str(result.num_trades),
            f"{result.backtest_duration_seconds:.1f}",
            result.status,
            result.description.replace("\t", " "),
        ]
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(row)

    def read_all(self) -> List[ExperimentResult]:
        """Read all logged experiments."""
        if not self.path.exists():
            return []
        results = []
        with open(self.path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                results.append(
                    ExperimentResult(
                        experiment_id=row.get("experiment_id", ""),
                        thesis_id=row.get("thesis_id", ""),
                        spec_id=row.get("spec_id", ""),
                        composite_score=float(row.get("composite_score", 0)),
                        sharpe_ratio=float(row.get("sharpe_ratio", 0)),
                        max_drawdown=float(row.get("max_drawdown", 0)),
                        hit_rate=float(row.get("hit_rate", 0)),
                        cumulative_return=float(row.get("cumulative_return", 0)),
                        num_trades=int(row.get("num_trades", 0)),
                        backtest_duration_seconds=float(
                            row.get("backtest_duration_seconds", 0)
                        ),
                        status=row.get("status", ""),
                        description=row.get("description", ""),
                    )
                )
        return results

    def best_score(self) -> float:
        """Return the best composite score from kept experiments."""
        results = self.read_all()
        kept = [r for r in results if r.status == "keep"]
        if not kept:
            return 0.0
        return max(r.composite_score for r in kept)

    def recent(self, n: int = 20) -> List[ExperimentResult]:
        """Return the N most recent experiments."""
        all_results = self.read_all()
        return all_results[-n:]

    def kept_count(self) -> int:
        return sum(1 for r in self.read_all() if r.status == "keep")

    def total_count(self) -> int:
        return len(self.read_all())

    def format_summary(self) -> str:
        """Human-readable summary like autoresearch's results.tsv display."""
        results = self.read_all()
        if not results:
            return "No experiments yet."
        kept = [r for r in results if r.status == "keep"]
        discarded = [r for r in results if r.status == "discard"]
        crashed = [r for r in results if r.status in ("crash", "timeout")]
        best = max(kept, key=lambda r: r.composite_score) if kept else None
        lines = [
            f"Total experiments: {len(results)}",
            f"  Kept: {len(kept)}  |  Discarded: {len(discarded)}  |  Crashed: {len(crashed)}",
        ]
        if best:
            lines.append(
                f"  Best: {best.composite_score:.6f} (Sharpe={best.sharpe_ratio:.2f}, "
                f"Return={best.cumulative_return:.4f}, DD={best.max_drawdown:.4f}) "
                f"[{best.description}]"
            )
        return "\n".join(lines)
