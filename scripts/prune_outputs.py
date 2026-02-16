#!/usr/bin/env python3
"""Prune bulky, low-value output artifacts while preserving research metadata.

This script is intentionally conservative:
- Keeps strategy code/params/docs untouched.
- Keeps evaluation summaries/leaderboards/candidate JSON files.
- Only removes heavy per-run payload files (e.g., decisions/equity traces).
- Optionally drops obvious dead-end search folders.

Run with --apply to actually delete files. Default mode is dry-run.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_HEAVY_FILES = ("decisions.jsonl", "equity_curve.csv")
TOKEN_RE = re.compile(r"(cand_\d+|candidate[_-]?\d+)", re.IGNORECASE)


@dataclass
class PruneStats:
    files_removed: int = 0
    dirs_removed: int = 0
    bytes_reclaimed: int = 0

    def add_file(self, size: int) -> None:
        self.files_removed += 1
        self.bytes_reclaimed += max(0, size)

    def add_dir(self) -> None:
        self.dirs_removed += 1


def _fmt_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(value)
    unit = 0
    while f >= 1024 and unit < len(units) - 1:
        f /= 1024.0
        unit += 1
    return f"{f:.2f}{units[unit]}"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _safe_unlink(path: Path, *, apply: bool) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if apply:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return 0
    return size


def _safe_rmtree(path: Path, *, apply: bool) -> bool:
    if apply:
        try:
            shutil.rmtree(path)
            return True
        except OSError:
            return False
    return True


def _list_job_dirs(parent: Path) -> list[Path]:
    if not parent.exists():
        return []
    return sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def _older_than(path: Path, cutoff: datetime) -> bool:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    return mtime < cutoff


def _extract_top_tokens(leaderboard_csv: Path, top_k: int) -> set[str]:
    tokens: set[str] = set()
    if not leaderboard_csv.exists() or top_k <= 0:
        return tokens
    try:
        with leaderboard_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= top_k:
                    break
                for value in row.values():
                    if not value:
                        continue
                    v = str(value)
                    for match in TOKEN_RE.findall(v):
                        tokens.add(match.lower())
    except Exception:
        return set()
    return tokens


def _is_protected_run_dir(path: Path, protected_tokens: set[str]) -> bool:
    if not protected_tokens:
        return False
    name = path.name.lower()
    if name in protected_tokens:
        return True
    return any(token in name for token in protected_tokens)


def _prune_heavy_files_in_dir(
    root: Path,
    *,
    heavy_names: set[str],
    apply: bool,
    stats: PruneStats,
    verbose: bool,
) -> None:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name not in heavy_names:
            continue
        size = _safe_unlink(p, apply=apply)
        if size > 0:
            stats.add_file(size)
            if verbose:
                action = "delete" if apply else "would-delete"
                print(f"{action}: {p} ({_fmt_bytes(size)})")


def _prune_job_dir(
    job_dir: Path,
    *,
    heavy_names: set[str],
    keep_top_runs: int,
    apply: bool,
    stats: PruneStats,
    verbose: bool,
) -> None:
    protected_tokens = _extract_top_tokens(job_dir / "leaderboard.csv", keep_top_runs)

    # Case 1: strategy_eval/<job>/runs/<candidate_or_run_id>/...
    runs_dir = job_dir / "runs"
    if runs_dir.exists() and runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if _is_protected_run_dir(run_dir, protected_tokens):
                continue
            _prune_heavy_files_in_dir(
                run_dir,
                heavy_names=heavy_names,
                apply=apply,
                stats=stats,
                verbose=verbose,
            )

    # Case 2: strategy_strict_gate_search/<job>/candidate_runs/cand_###/wXX/... files
    candidate_runs = job_dir / "candidate_runs"
    if candidate_runs.exists() and candidate_runs.is_dir():
        for cand_dir in candidate_runs.iterdir():
            if not cand_dir.is_dir():
                continue
            if _is_protected_run_dir(cand_dir, protected_tokens):
                continue
            _prune_heavy_files_in_dir(
                cand_dir,
                heavy_names=heavy_names,
                apply=apply,
                stats=stats,
                verbose=verbose,
            )


def _is_deadend_search_dir(path: Path) -> bool:
    """Heuristic: folder has no leaderboard/candidates and only tiny config crumbs."""
    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        return True
    names = {p.name for p in files}
    has_signal = any(
        n in names for n in ("leaderboard.csv", "leaderboard.json", "evaluation_result.json", "candidates.json")
    )
    has_candidates_dir = (path / "candidates").exists()
    if has_signal or has_candidates_dir:
        return False
    if names.issubset({"search_config.json", "summary.json", "README.txt", ".DS_Store"}):
        return True
    # Very tiny folders without useful outputs are considered dead-ends.
    return _dir_size(path) < 32 * 1024


def _remove_deadend_dirs(
    parent: Path,
    *,
    keep_recent_dirs: int,
    cutoff: datetime,
    apply: bool,
    stats: PruneStats,
    verbose: bool,
) -> None:
    job_dirs = _list_job_dirs(parent)
    keep = {p.name for p in job_dirs[: max(0, keep_recent_dirs)]}
    for job_dir in job_dirs:
        if job_dir.name in keep:
            continue
        if not _older_than(job_dir, cutoff):
            continue
        if not _is_deadend_search_dir(job_dir):
            continue
        size = _dir_size(job_dir)
        ok = _safe_rmtree(job_dir, apply=apply)
        if ok:
            stats.bytes_reclaimed += size
            stats.add_dir()
            if verbose:
                action = "delete-dir" if apply else "would-delete-dir"
                print(f"{action}: {job_dir} ({_fmt_bytes(size)})")


def _prune_parent(
    parent: Path,
    *,
    keep_recent_dirs: int,
    keep_top_runs: int,
    cutoff: datetime,
    heavy_names: set[str],
    drop_deadends: bool,
    apply: bool,
    stats: PruneStats,
    verbose: bool,
) -> None:
    job_dirs = _list_job_dirs(parent)
    keep = {p.name for p in job_dirs[: max(0, keep_recent_dirs)]}
    for job_dir in job_dirs:
        if job_dir.name in keep:
            continue
        if not _older_than(job_dir, cutoff):
            continue
        _prune_job_dir(
            job_dir,
            heavy_names=heavy_names,
            keep_top_runs=keep_top_runs,
            apply=apply,
            stats=stats,
            verbose=verbose,
        )
    if drop_deadends:
        _remove_deadend_dirs(
            parent,
            keep_recent_dirs=keep_recent_dirs,
            cutoff=cutoff,
            apply=apply,
            stats=stats,
            verbose=verbose,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune stale heavy output artifacts safely.")
    parser.add_argument("--root", type=Path, default=Path("outputs"), help="Output root directory.")
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=8.0,
        help="Only prune directories older than this many hours.",
    )
    parser.add_argument(
        "--keep-recent-dirs",
        type=int,
        default=15,
        help="Keep this many newest job directories under each parent untouched.",
    )
    parser.add_argument(
        "--keep-top-runs",
        type=int,
        default=3,
        help="Preserve heavy payload files for top N leaderboard entries per job.",
    )
    parser.add_argument(
        "--heavy-file",
        action="append",
        default=[],
        help=f"Additional heavy filename to prune (default: {', '.join(DEFAULT_HEAVY_FILES)}).",
    )
    parser.add_argument(
        "--drop-deadends",
        action="store_true",
        help="Remove obvious dead-end search dirs (tiny/no leaderboard/candidates).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each deletion candidate.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files/directories.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=float(args.min_age_hours))
    heavy_names = set(DEFAULT_HEAVY_FILES)
    heavy_names.update([name.strip() for name in args.heavy_file if name.strip()])

    parents: Iterable[Path] = (
        root / "evaluations" / "strategy_eval",
        root / "evaluations" / "strategy_strict_gate_search",
        root / "backtests",
    )

    stats = PruneStats()
    for parent in parents:
        _prune_parent(
            parent,
            keep_recent_dirs=int(args.keep_recent_dirs),
            keep_top_runs=int(args.keep_top_runs),
            cutoff=cutoff,
            heavy_names=heavy_names,
            drop_deadends=bool(args.drop_deadends),
            apply=bool(args.apply),
            stats=stats,
            verbose=bool(args.verbose),
        )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"[{mode}] files={stats.files_removed} dirs={stats.dirs_removed} "
        f"reclaimed={_fmt_bytes(stats.bytes_reclaimed)}"
    )
    if not args.apply:
        print("No files were deleted. Re-run with --apply to execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
