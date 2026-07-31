"""Pure command-line outcome policy for collection runs."""

from __future__ import annotations

from typing import Any, Mapping


_OPERATIONAL_FAILURES = {
    "aborted",
    "invalid_task_configuration",
    "no_samples",
    "recorder_error",
    "runtime_error",
}


def collection_exit_code(
    summary: Mapping[str, Any],
    *,
    require_all_success: bool,
) -> int:
    """Separate a completed collection from task and infrastructure outcomes."""

    requested = summary.get("requested_episodes")
    reports = summary.get("episodes")
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or not isinstance(reports, list)
        or len(reports) != requested
    ):
        return 1
    if any(
        not isinstance(report, Mapping)
        or report.get("failure_reason") in _OPERATIONAL_FAILURES
        for report in reports
    ):
        return 1
    successful = summary.get("successful_episodes")
    if (
        require_all_success
        and (
            not isinstance(successful, int)
            or isinstance(successful, bool)
            or successful != requested
        )
    ):
        return 3
    return 0
