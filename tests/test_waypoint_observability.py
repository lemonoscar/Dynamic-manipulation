from scripts.probe_waypoint_observability import balanced_indices, transition_labels


def _row(episode, row, timestamp, route, prefix):
    return {
        "source_episode_id": episode,
        "source_row_id": row,
        "timestamp": timestamp,
        "route": route,
        "action_valid_mask": [index < prefix for index in range(20)],
    }


def test_transition_labels_keep_episode_boundaries_and_original_prefix():
    records = [
        _row("a", 0, 0.0, "NAV_TO_SOURCE", 20),
        _row("a", 1, 0.2, "NAV_TO_SOURCE", 3),
        _row("a", 2, 0.4, "PICK", 20),
        _row("a", 3, 1.6, "PICK", 20),
        _row("b", 0, 0.0, "PLACE", 1),
        _row("b", 1, 0.2, "DONE", 0),
    ]

    labels = transition_labels(records)

    assert labels[1]["boundary_class"] == "BEFORE"
    assert labels[1]["transition"] == "NAV_TO_SOURCE->PICK"
    assert labels[1]["time_to_boundary_s"] == 0.2
    assert labels[1]["original_valid_prefix_k"] == 3
    assert labels[2]["boundary_class"] == "AFTER"
    assert labels[2]["transition"] == "NAV_TO_SOURCE->PICK"
    assert labels[3]["boundary_class"] == "INTERIOR"
    assert labels[4]["transition"] == "PLACE->DONE"
    assert labels[5]["source_episode_id"] == "b"


def test_balanced_indices_are_deterministic_and_cover_rare_buckets():
    rows = [
        {
            "route": route,
            "boundary_class": boundary,
            "transition": transition,
        }
        for route, boundary, transition in (
            ("NAV_TO_SOURCE", "INTERIOR", None),
            ("NAV_TO_SOURCE", "INTERIOR", None),
            ("NAV_TO_SOURCE", "BEFORE", "NAV_TO_SOURCE->PICK"),
            ("PICK", "AFTER", "NAV_TO_SOURCE->PICK"),
            ("PLACE", "BEFORE", "PLACE->DONE"),
        )
    ]

    first = balanced_indices(rows, 4, 17)
    second = balanced_indices(rows, 4, 17)

    assert first == second
    assert len(set(first)) == 4
    assert 2 in first and 3 in first and 4 in first
