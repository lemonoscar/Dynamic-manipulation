from conveyor_bench.cli import collection_exit_code


def _summary(*failure_reasons: str) -> dict:
    reports = [
        {
            "success": reason == "none",
            "failure_reason": reason,
        }
        for reason in failure_reasons
    ]
    return {
        "requested_episodes": len(reports),
        "successful_episodes": sum(report["success"] for report in reports),
        "episodes": reports,
    }


def test_completed_task_failures_are_valid_collection_output():
    assert (
        collection_exit_code(
            _summary("none", "target_missed", "dropped"),
            require_all_success=False,
        )
        == 0
    )


def test_strict_gate_distinguishes_task_failure_from_cli_error():
    assert (
        collection_exit_code(
            _summary("none", "target_missed"),
            require_all_success=True,
        )
        == 3
    )
    assert collection_exit_code(_summary("none"), require_all_success=True) == 0


def test_operational_failure_is_nonzero():
    assert (
        collection_exit_code(
            _summary("runtime_error"),
            require_all_success=False,
        )
        == 1
    )


def test_incomplete_run_is_nonzero():
    summary = _summary("none")
    summary["requested_episodes"] = 2
    assert collection_exit_code(summary, require_all_success=False) == 1
