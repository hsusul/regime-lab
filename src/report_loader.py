"""Read-only helpers for loading the latest generated analysis reports.

This module never triggers report generation. It scans reports/ for existing
JSON files and returns the newest match by filename timestamp.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.train import REPORTS_DIR


class ReportNotFoundError(Exception):
    """Raised when no matching report file exists."""


# Filename prefixes and glob patterns used by each analysis module.
_REPORT_PATTERNS: dict[str, str] = {
    "hmm": "hmm_*_report.json",
    "forward_returns": "forward_returns_*.json",
    "walk_forward": "walk_forward_*.json",
    "model_comparison": "model_comparison_*.json",
    "explain": "explain_*.json",
}


def latest_report(
    report_type: str,
    *,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Load the newest JSON report matching *report_type*.

    Parameters
    ----------
    report_type:
        One of ``hmm``, ``forward_returns``, ``walk_forward``,
        ``model_comparison``, or ``explain``.
    reports_dir:
        Override for the reports directory (used in tests).

    Returns
    -------
    dict
        The parsed JSON contents of the most recent report file.

    Raises
    ------
    ReportNotFoundError
        If *report_type* is unknown or no matching file exists.
    """
    pattern = _REPORT_PATTERNS.get(report_type)
    if pattern is None:
        raise ReportNotFoundError(
            f"Unknown report type {report_type!r}. "
            f"Expected one of: {', '.join(sorted(_REPORT_PATTERNS))}"
        )

    selected_dir = reports_dir or REPORTS_DIR
    matches = sorted(selected_dir.glob(pattern))
    if not matches:
        raise ReportNotFoundError(
            f"No {report_type} report found in {selected_dir}."
        )

    newest = matches[-1]
    with newest.open("r", encoding="utf-8") as fh:
        return json.load(fh)
