import json

import pytest

from carlautopilot.core import Action, EpisodeHeader, Event, EventKind, GroundTruth, StepRecord
from carlautopilot.eval import EpisodeLogWriter, aggregate, compute_metrics, make_report, read_episode

HEADER = EpisodeHeader(scenario_name="직진", seed=3, dt=0.5, policy_name="p")


def rec(step, speed, progress, length=100.0, events=(), lead=None, t=None):
    truth = GroundTruth(x=progress, y=0.0, yaw_rad=0.0, speed_mps=speed, route_progress_m=progress, route_length_m=length, lateral_error_m=0.0)
    return StepRecord(step=step, t=step * 0.5 if t is None else t, speed_mps=speed, command=0, action=Action(0.1, 0.2, 0.0), truth=truth, events=tuple(events), lead_distance_m=lead)


def full_run(events_at=None):
    out = []
    for i in range(5):
        ev = events_at.get(i, ()) if events_at else ()
        out.append(rec(i, 10.0, i * 25.0, events=ev))
    return out


def test_roundtrip(tmp_path):
    p = tmp_path / "ep.jsonl"
    records = [rec(0, 1.0, 0.0, lead=None), rec(1, 2.0, 1.0, events=[Event(0.5, EventKind.RED_LIGHT, "x")], lead=12.5)]
    with EpisodeLogWriter(p) as w:
        w.write_header(HEADER)
        for r in records:
            w.write(r)
    header, back = read_episode(p)
    assert header == HEADER
    assert back == records
    assert back[0].lead_distance_m is None and back[1].lead_distance_m == 12.5


def test_overwrite_protection_and_order(tmp_path):
    p = tmp_path / "ep.jsonl"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        EpisodeLogWriter(p)
    EpisodeLogWriter(p, overwrite=True)
    with EpisodeLogWriter(tmp_path / "new.jsonl") as w:
        with pytest.raises(RuntimeError):
            w.write(rec(0, 1.0, 0.0))
        w.write_header(HEADER)
        with pytest.raises(RuntimeError):
            w.write_header(HEADER)


def test_bad_schema_and_empty(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"scenario_name": "a", "seed": 1, "dt": 0.1, "policy_name": "p", "schema_version": 99}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_episode(p)
    e = tmp_path / "empty.jsonl"
    e.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        read_episode(e)


def test_full_completion_score_one():
    m = compute_metrics(HEADER, full_run(), "route_complete")
    assert m["route_completion"] == 1.0 and m["driving_score"] == 1.0 and m["success"] is True


def test_half_progress():
    recs = [rec(i, 5.0, i * 12.5) for i in range(5)]
    m = compute_metrics(HEADER, recs, "timeout")
    assert m["route_completion"] == pytest.approx(0.5)
    assert m["driving_score"] == pytest.approx(0.5)
    assert m["success"] is False


def test_penalties_multiply():
    ev1 = [Event(0.0, EventKind.RED_LIGHT)]
    m = compute_metrics(HEADER, full_run({1: ev1}), "route_complete")
    assert m["driving_score"] == pytest.approx(0.7)
    two = compute_metrics(HEADER, full_run({1: ev1, 2: [Event(1.0, EventKind.STOP_SIGN)]}), "route_complete")
    assert two["driving_score"] == pytest.approx(0.7 * 0.8)
    same_twice = compute_metrics(HEADER, full_run({1: ev1, 2: ev1}), "route_complete")
    assert same_twice["driving_score"] == pytest.approx(0.49)
    assert same_twice["events"] == {EventKind.RED_LIGHT: 2}


def test_collision_is_not_success_and_custom_penalty():
    ev = [Event(0.0, EventKind.COLLISION_VEHICLE)]
    m = compute_metrics(HEADER, full_run({4: ev}), "route_complete", penalties={EventKind.COLLISION_VEHICLE: 0.1})
    assert m["success"] is False
    assert m["driving_score"] == pytest.approx(0.1)


def test_jerk_decel_lead_duration():
    speeds = [10.0, 10.0, 8.0, 8.0]
    recs = [rec(i, v, i * 5.0, lead=[None, 30.0, 12.0, 20.0][i]) for i, v in enumerate(speeds)]
    m = compute_metrics(HEADER, recs, "timeout")
    assert m["max_decel_mps2"] == pytest.approx(4.0)
    assert m["mean_abs_jerk"] == pytest.approx((8.0 + 8.0) / 2)
    assert m["min_lead_distance_m"] == 12.0
    assert m["duration_s"] == pytest.approx(1.5)
    assert m["mean_speed_mps"] == pytest.approx(9.0)
    assert compute_metrics(HEADER, [rec(0, 1.0, 0.0)], "timeout")["min_lead_distance_m"] is None


def test_no_records():
    with pytest.raises(ValueError):
        compute_metrics(HEADER, [], None)


def test_aggregate_and_report():
    a = compute_metrics(HEADER, full_run(), "route_complete")
    b = compute_metrics(HEADER, full_run({1: [Event(0.0, EventKind.RED_LIGHT)]}), "route_complete")
    agg = aggregate([a, b])
    assert agg["episodes"] == 2
    assert agg["mean_driving_score"] == pytest.approx(0.85)
    assert agg["min_driving_score"] == pytest.approx(0.7)
    assert agg["success_rate"] == 1.0
    assert agg["events"] == {EventKind.RED_LIGHT: 1}
    text = make_report(agg, [a, b], title="T")
    assert text.startswith("# T") and "| scenario |" in text and "0.850" in text and "red_light=1" in text
    with pytest.raises(ValueError):
        aggregate([])
