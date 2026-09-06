"""Physical-unit metrics and episode-clustered uncertainty; no model imports."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np

from .joint_trajectory import JointTrajectoryRoute, action_domain, JointTrajectoryDomain
from .joint_trajectory_runtime import DirectJointTrajectoryExecutor, JointSafetyLimits, RouteCommitter

ROUTES = [r.value for r in JointTrajectoryRoute]
LIMITS = JointSafetyLimits((-2.618, 0., 0., -1.5708, -1.5708, -1.5708),
                          (3.14, 3.14, 3.14, 1.5708, 1.5708, 1.5708), (3.,) * 6)


def saturation_gate(metric):
    value = metric.get('sample_mean')
    return {'rate':value, 'limit':.005,
            'passed':None if value is None else value <= .005,
            'definition':'(position + rate + gripper clipping events) / (samples * 10 * 7); events may overlap'}


def trajectory_metrics(route, predicted, target, state, real_points=10):
    nav = action_domain(route) is JointTrajectoryDomain.NAVIGATION
    width = 3 if nav else 7
    p, t = np.asarray(predicted, dtype=float), np.asarray(target, dtype=float)
    if p.shape != (10, width) or t.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("trajectory shape or finiteness invalid")
    if not 1 <= real_points <= 10:
        raise ValueError("real future point count must be 1..10")
    result = {}
    if nav:
        distance = np.linalg.norm(p[:, :2] - t[:, :2], axis=1)
        yaw = np.abs((p[:, 2] - t[:, 2] + math.pi) % (2 * math.pi) - math.pi)
        result.update(xy_ade_m=float(distance.mean()), xy_fde_m=float(distance[-1]),
                      yaw_mae_rad=float(yaw.mean()), yaw_final_rad=float(yaw[-1]))
        error = distance
        for h in range(10):
            result[f"xy_h{h+1:02d}_m"] = float(distance[h])
    else:
        q = np.asarray(state[:6], dtype=float)
        joint = np.abs(p[:, :6] - t[:, :6])
        result.update(joint_mae_rad=float(joint.mean()), joint_final_l2_rad=float(np.linalg.norm(p[-1, :6] - t[-1, :6])),
                      gripper_mae=float(np.abs(p[:, 6] - t[:, 6]).mean()),
                      gripper_binary_accuracy=float(((p[:, 6] >= .5) == (t[:, 6] >= .5)).mean()))
        for axis in range(6):
            result[f"joint_{axis+1}_mae_rad"] = float(joint[:, axis].mean())
        for h in range(10):
            result[f"joint_h{h+1:02d}_mae_rad"] = float(joint[h].mean())
        chunk = DirectJointTrajectoryExecutor(LIMITS).prepare(q, p)
        target_chunk = DirectJointTrajectoryExecutor(LIMITS).prepare(q, t)
        applied = np.array([[*c.joint_position, c.gripper_open_fraction] for c in chunk.commands])
        raw = p.copy()
        raw[:, :6] += q
        result.update(saturation_rate=chunk.saturation_rate,
                      target_saturation_rate=target_chunk.saturation_rate,
                      position_saturation_rate=chunk.position_saturation_count / 60,
                      rate_saturation_rate=chunk.rate_saturation_count / 60,
                      gripper_saturation_rate=chunk.gripper_saturation_count / 10,
                      unique_saturation_fraction=float((np.abs(applied - raw) > 1e-12).mean()),
                      executed_joint_mae_rad=float(np.abs(applied[:, :6] - (t[:, :6] + q)).mean()))
        error = joint.mean(axis=1)
    result["real_future_error"] = float(error[:real_points].mean())
    result["hold_tail_error"] = float(error[real_points:].mean()) if real_points < 10 else None
    return result


def cluster_mean(values, episodes, *, seed=20260905, draws=2000):
    """Mean/CI with entire episodes resampled (repeats within episode stay together)."""
    binary_observations = all(isinstance(v, (bool, np.bool_)) for v in values if v is not None)
    grouped = defaultdict(list)
    for value, episode in zip(values, episodes, strict=True):
        if value is not None:
            if not math.isfinite(float(value)):
                raise ValueError("nonfinite metric")
            grouped[episode].append(float(value))
    if not grouped:
        return {"sample_mean": None, "episode_mean": None, "ci95": None, "episodes": 0, "samples": 0}
    groups = [grouped[key] for key in sorted(grouped)]
    means = np.array([np.mean(g) for g in groups])
    result = {"sample_mean": float(np.mean([v for g in groups for v in g])),
              "episode_mean": float(means.mean()), "episodes": len(groups),
              "samples": sum(map(len, groups)), "ci95": None}
    if len(groups) > 1:
        rng = np.random.default_rng(seed)
        bootstrap = means[rng.integers(len(groups), size=(draws, len(groups)))].mean(axis=1)
        result["ci95"] = np.quantile(bootstrap, [.025, .975]).tolist()
        result["ci_method"] = "episode_cluster_percentile_bootstrap"
        if binary_observations and (np.all(means == 0) or np.all(means == 1)):
            # A degenerate bootstrap [1,1] is not evidence of certain success.
            z2 = 1.959963984540054 ** 2
            n, value = len(means), float(means[0])
            center = (value + z2 / (2*n)) / (1 + z2/n)
            half = (z2**.5 * (value*(1-value)/n + z2/(4*n*n))**.5) / (1+z2/n)
            result["ci95"] = [max(0., center-half), min(1., center+half)]
            result["ci_method"] = "Wilson_boundary_envelope_using_episode_count"
    return result


def summarize(rows):
    if not rows:
        raise ValueError("cannot summarize no evaluation rows")
    episodes = [r["episode_id"] for r in rows]
    confusion = {a: {b: 0 for b in [*ROUTES, "RECOVER"]} for a in ROUTES}
    for row in rows:
        confusion[row["target_route"]][row["predicted_route"]] += 1
    per_route = {}
    for route in ROUTES:
        tp = confusion[route][route]
        count = sum(confusion[route].values())
        predicted = sum(confusion[r][route] for r in ROUTES)
        precision = tp / predicted if predicted else 0.
        recall = tp / count if count else None
        f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.
        per_route[route] = {"count": count, "precision": precision, "recall": recall, "f1": f1}
    result = {"rows": len(rows), "episodes": len(set(episodes)), "confusion": confusion,
              "per_route": per_route, "macro_f1": float(np.mean([v["f1"] for v in per_route.values()])),
              "route_accuracy": cluster_mean([r["route_correct"] for r in rows], episodes),
              "raw_route_accuracy": cluster_mean([r["raw_route_correct"] for r in rows], episodes),
              "invalid_rate": cluster_mean([not r["route_valid"] for r in rows], episodes),
              "action_coverage": cluster_mean([r.get("predicted") is not None for r in rows], episodes),
              "failure_counts": dict(Counter(r["action_failure"] for r in rows if r["action_failure"])),
              "invalid_reasons": dict(Counter(r["recover_reason"] for r in rows if r["recover_reason"])),
              "actions": {}}
    for mode in ("predicted", "oracle", "baseline"):
        result["actions"][mode] = {}
        for route in [*ROUTES, "NAVIGATION", "MANIPULATION"]:
            selected = [r for r in rows if r.get(mode) is not None and
                        (r["target_route"] == route or action_domain(r["target_route"]).value == route)]
            keys = sorted({k for r in selected for k in r[mode]})
            result["actions"][mode][route] = {k: cluster_mean([r[mode].get(k) for r in selected],
                                                            [r["episode_id"] for r in selected]) for k in keys}
    result["strata"] = {}
    for name, predicate in (("boundary", lambda r: r["transition_window"]),
                            ("interior", lambda r: not r["transition_window"]),
                            ("gripper_transition", lambda r: r["gripper_transition"]),
                            ("terminal_hold", lambda r: r["real_future_points"] < 10)):
        selected = [r for r in rows if predicate(r)]
        result["strata"][name] = {"rows": len(selected), "route_accuracy": cluster_mean(
            [r["route_correct"] for r in selected], [r["episode_id"] for r in selected])}
    events = defaultdict(list)
    for r in rows:
        if r["transition_id"]:
            events[(r["episode_id"], r.get("diffusion_seed", 17), r["transition_id"])].append(r)
    boundary = defaultdict(list)
    for event in events.values():
        before = [r for r in event if r["boundary_signed_time_s"] < 0]
        after = [r for r in event if r["boundary_signed_time_s"] >= 0]
        if before and after:
            b = max(before, key=lambda r: r["boundary_signed_time_s"])
            a = min(after, key=lambda r: r["boundary_signed_time_s"])
            old, new = a["boundary_transition"].split("->")
            increase = (a["route_probs"][new] - a["route_probs"][old]) - (b["route_probs"][new] - b["route_probs"][old])
            boundary[a["boundary_transition"]].append({"episode_id": a["episode_id"], "margin_increase": increase})
    result["boundary_pairs"] = {key: {"events": len(items), "positive_margin_change": cluster_mean(
        [r["margin_increase"] > 0 for r in items], [r["episode_id"] for r in items])} for key, items in boundary.items()}
    sequences = defaultdict(list)
    for r in rows:
        if "query_timestamp_s" in r:
            sequences[(r["episode_id"], r["diffusion_seed"])].append(r)
    timing = defaultdict(list)
    for (episode, _seed), sequence in sequences.items():
        sequence.sort(key=lambda r: r["query_timestamp_s"])
        committer, previous, switches = RouteCommitter(), None, []
        for i, r in enumerate(sequence):
            if not r["route_valid"]:
                continue
            committed = committer.observe(r["route_probs"], sequence_id=i).committed_route
            if committed is not None and committed != previous:
                switches.append((committed.value, r["query_timestamp_s"]))
                previous = committed
        event_times = {}
        for r in sequence:
            if r["transition_id"]:
                event_times[r["boundary_transition"]] = r["query_timestamp_s"] - r["boundary_signed_time_s"]
        for transition, timestamp in event_times.items():
            new = transition.split("->")[1]
            candidates = [t for route, t in switches if route == new and abs(t-timestamp) <= 2.]
            delay = min(candidates, key=lambda t: abs(t-timestamp)) - timestamp if candidates else None
            timing[transition].append((episode, delay))
    result["offline_route_confirmation"] = {transition: {
        "transition_delay_s": cluster_mean([delay for _, delay in items], [episode for episode, _ in items]),
        "missed_within_2s": sum(delay is None for _, delay in items),
        "note": "Teacher-recorded observations, not a physical closed-loop rollout; negative delay means early switch."}
        for transition, items in timing.items()}
    result["saturation_gate"] = {}
    for mode in ("predicted", "oracle"):
        metric = result["actions"][mode]["MANIPULATION"].get("saturation_rate", {})
        result["saturation_gate"][mode] = saturation_gate(metric)
    return result
