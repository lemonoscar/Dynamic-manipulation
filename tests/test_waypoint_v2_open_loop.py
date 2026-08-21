from scripts import evaluate_waypoint_v2_open_loop as evaluation


def test_synchronized_batches_keep_all_zero3_ranks_on_identical_examples():
    selected = list(range(64))
    batches = evaluation._synchronized_batches(
        selected, per_rank_batch_size=2, world_size=4
    )
    assert len(batches) == 8
    assert all(len(batch) == 8 for batch in batches)
    assert [index for batch in batches for index in batch] == selected


def _dataset():
    routes = []
    boundaries = []
    transition_ids = []
    progress = []
    transitions = (
        ("NAV_TO_SOURCE", "PICK"),
        ("PICK", "NAV_TO_TARGET"),
        ("NAV_TO_TARGET", "PLACE"),
        ("PLACE", "DONE"),
    )
    for event_index, (old, new) in enumerate(transitions):
        name = f"{old}->{new}"
        for row in range(11):
            routes.append(old if row < 5 else new)
            boundaries.append(name)
            transition_ids.append(f"episode:{event_index}:{name}")
            progress.append(row / 10.0)
    for repeat in range(8):
        for route_index, route in enumerate(
            ("NAV_TO_SOURCE", "PICK", "NAV_TO_TARGET", "PLACE", "DONE")
        ):
            routes.append(route)
            boundaries.append(None)
            transition_ids.append(None)
            progress.append((repeat + route_index / 5.0) / 8.0)
    return type(
        "Dataset",
        (),
        {
            "routes": routes,
            "boundaries": boundaries,
            "transition_ids": transition_ids,
            "phase_progress": progress,
        },
    )()


def test_transition_selection_covers_full_windows_routes_and_boundaries():
    dataset = _dataset()
    selected = evaluation._transition_centric_selection(
        dataset, list(range(len(dataset.routes))), 64
    )
    assert len(selected) == len(set(selected)) == 64
    assert {dataset.routes[index] for index in selected} == {
        "NAV_TO_SOURCE",
        "PICK",
        "NAV_TO_TARGET",
        "PLACE",
        "DONE",
    }
    assert {
        dataset.boundaries[index]
        for index in selected
        if dataset.boundaries[index] is not None
    } == {
        "NAV_TO_SOURCE->PICK",
        "PICK->NAV_TO_TARGET",
        "NAV_TO_TARGET->PLACE",
        "PLACE->DONE",
    }


def test_transition_metrics_report_lag_crossover_flicker_and_interior():
    rows = []
    for index, signed_time in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
        new_probability = (signed_time + 1.0) / 2.0
        rows.append(
            {
                "transition_id": "episode:NAV_TO_SOURCE->PICK",
                "boundary_signed_time_s": signed_time,
                "boundary_transition": "NAV_TO_SOURCE->PICK",
                "source_episode_id": "episode",
                "sample_id": f"episode:{index}",
                "target": "NAV_TO_SOURCE" if signed_time < 0 else "PICK",
                "predicted": "NAV_TO_SOURCE" if signed_time < 0 else "PICK",
                "route_probs": {
                    "NAV_TO_SOURCE": 1.0 - new_probability,
                    "PICK": new_probability,
                },
                "transition_window": True,
            }
        )
    rows.extend(
        {
            "transition_id": None,
            "boundary_signed_time_s": None,
            "boundary_transition": None,
            "source_episode_id": "episode",
            "sample_id": f"interior:{index}",
            "target": route,
            "predicted": route,
            "route_probs": {},
            "transition_window": False,
        }
        for index, route in enumerate(
            ("NAV_TO_SOURCE", "PICK", "NAV_TO_TARGET", "PLACE", "DONE")
        )
    )
    report = evaluation._transition_metrics(rows)
    event = report["events"][0]
    assert event["early_switch_rate"] == 0.0
    assert event["late_switch_rate"] == 0.0
    assert event["switch_lag_s"] == 0.0
    assert event["switch_lag_queries"] == 0
    assert event["logit_crossover_s"] == 0.0
    assert event["flicker_count"] == 0
    assert report["phase_interior_macro_accuracy"] == 1.0


def test_prefix_metrics_separate_overrun_and_underrun():
    rows = [
        {
            "target": "NAV_TO_SOURCE",
            "predicted_prefix_k": 8,
            "prefix_target_k": 6,
        },
        {
            "target": "PICK",
            "predicted_prefix_k": 4,
            "prefix_target_k": 6,
        },
        {
            "target": "DONE",
            "predicted_prefix_k": None,
            "prefix_target_k": 0,
        },
    ]
    report = evaluation._prefix_metrics(rows)
    assert report["mae_k"] == 2.0
    assert report["overrun_rate"] == 0.5
    assert report["mean_under_run_points"] == 1.0
    assert report["mean_over_run_points"] == 1.0
