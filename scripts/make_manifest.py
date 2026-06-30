"""
Generate ``results_manifest.json`` — a single provenance record for every number the
project cites. It does NOT recompute anything; it reads the committed eval artifacts
under ``eval/results/`` plus the load-test report, and stamps the environment (git SHA,
platform, key package versions). Run it after ``make reproduce`` so the manifest always
reflects the artifacts on disk.

Run:  python -m scripts.make_manifest   (or: make manifest)
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval" / "results"
LOAD_REPORT = REPO_ROOT / "tests" / "load" / "results" / "load_report.md"
OUT = REPO_ROOT / "results_manifest.json"

# Metric keys worth surfacing per retriever (others are kept in the raw files).
_METRIC_KEYS = ("recall@1", "recall@5", "recall@5_ci95", "recall@10", "mrr", "ndcg@10")


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - manifest must not fail if git is unavailable
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _environment() -> Dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _pkg_version(name)
            for name in ("torch", "open_clip_torch", "qdrant-client", "sentence-transformers")
        },
    }


def _summarize_eval(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    summary: Dict[str, Any] = {
        k: data[k]
        for k in ("dataset", "task", "protocol", "gallery_size", "num_queries")
        if k in data
    }
    results = data.get("results")
    if isinstance(results, dict):
        per_retriever = {}
        for name, metrics in results.items():
            if isinstance(metrics, dict):
                per_retriever[name] = {k: metrics[k] for k in _METRIC_KEYS if k in metrics}
        if per_retriever:
            summary["results"] = per_retriever
    else:
        # Non-standard artifact (e.g. faithfulness/ablation): keep small scalar fields.
        summary.update({k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))})
    return summary


def _parse_load_report() -> Optional[List[Dict[str, str]]]:
    if not LOAD_REPORT.exists():
        return None
    rows = []
    for line in LOAD_REPORT.read_text().splitlines():
        # Match the results table rows: "| 500 users | 33,378 | 0 (0.00%) | ... |"
        if re.match(r"\|\s*\d+\s*users\s*\|", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append({"concurrency": cells[0], "raw": " | ".join(cells[1:])})
    return rows or None


def main() -> None:
    eval_files = sorted(RESULTS_DIR.glob("*.json"))
    manifest = {
        "description": "Provenance for every number cited by this project. "
        "Regenerate artifacts with `make reproduce`, then `make manifest`.",
        "environment": _environment(),
        "eval_artifacts": {p.name: _summarize_eval(p) for p in eval_files},
        "load_test": _parse_load_report(),
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"  git_sha={manifest['environment']['git_sha']}  eval_artifacts={len(eval_files)}")


if __name__ == "__main__":
    main()
