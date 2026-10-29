import math

import pytest

from carlautopilot.carla_env.route import WindowedRoute, carla_to_xy, carla_yaw_to_rad, commands_from_options, dedupe_points, junction_turn_options
from carlautopilot.core import Command


def test_carla_frame_flips_y_and_yaw():
    assert carla_to_xy(3.0, 4.0) == (3.0, -4.0)
    # CARLA yaw +90 points to +y (right-hand side of +x); project frame: -90 deg
    assert carla_yaw_to_rad(90.0) == pytest.approx(-math.pi / 2)


def test_commands_start_before_junction_and_return_to_follow():
    s = [0, 10, 20, 30, 40, 50, 60, 70]
    opts = [4, 4, 4, 1, 1, 4, 3, 4]
    cmds = commands_from_options(s, opts, lead_m=10.0)
    assert [(c.s_m, c.command) for c in cmds] == [
        (20, Command.LEFT),
        (40, Command.FOLLOW),
        (50, Command.STRAIGHT),
        (60, Command.FOLLOW),
    ]
    assert commands_from_options(s, [5, 6, 4, 4, 4, 4, 4, 4]) == ()


def test_windowed_projection_ignores_far_part_of_a_self_crossing_route():
    # out along x, back along a parallel line 1 m to the left
    r = WindowedRoute([(0, 0), (100, 0), (100, 1), (0, 1)])
    s_global, _, _ = r.project(50.0, 0.9)
    assert s_global > 100
    s, lat, dist = r.project(50.0, 0.9, 40.0, 60.0)
    assert s == pytest.approx(50.0)
    assert lat == pytest.approx(0.9)
    assert dist == pytest.approx(0.9)


def test_dedupe_drops_repeated_points():
    assert dedupe_points([(0, 0), (0, 0.01), (1, 0), (1, 0)]) == [0, 2]


def test_view_shift_makes_the_expert_steer_back_toward_the_route():
    from carlautopilot.carla_env.route import points_seen_from
    from carlautopilot.policies import PathFollowPolicy

    straight = [(d, 0.0) for d in (3.0, 6.0, 9.0, 12.0, 15.0, 20.0)]
    pol = PathFollowPolicy(max_steer_rad=1.22)
    left_shift = points_seen_from(straight, 1.0, 0.0)
    assert all(y == pytest.approx(-1.0) for _, y in left_shift)
    assert pol.act(_obs(left_shift)).steer > 0.02
    assert pol.act(_obs(points_seen_from(straight, -1.0, 0.0))).steer < -0.02
    assert pol.act(_obs(points_seen_from(straight, 0.0, 10.0))).steer > 0.02
    assert points_seen_from([(10.0, 0.0)], 0.0, 90.0)[0] == pytest.approx((0.0, -10.0))


def _obs(points):
    from carlautopilot.core import Observation

    return Observation(t=0.0, speed_mps=7.0, command=Command.FOLLOW, extras={"lookahead_points": points})


def test_junction_turn_options_relabels_a_planner_follow_turn():
    # straight east 20 m, a junction curving 90 deg left (north), straight north 20 m; planner said LANEFOLLOW (4)
    pts = [(x, 0.0) for x in range(0, 21, 2)] + [(20 + 10 * math.sin(a), 10 - 10 * math.cos(a)) for a in [k * math.pi / 16 for k in range(1, 8)]]
    pts += [(30.0, y) for y in range(10, 31, 2)]
    junc = [10 < k < 11 + 7 for k in range(len(pts))]
    opts = junction_turn_options(pts, [4] * len(pts), junc)
    assert {opts[k] for k in range(len(pts)) if junc[k]} == {1} and {opts[k] for k in range(len(pts)) if not junc[k]} == {4}
    assert junction_turn_options([(p[0], -p[1]) for p in pts], [4] * len(pts), junc)[12] == 2  # mirrored: right
    assert junction_turn_options(pts, [3] * len(pts), junc)[12] == 1  # STRAIGHT on a 90 deg turn is fixed too
    flat = [(float(x), 0.0) for x in range(0, 60, 2)]
    assert junction_turn_options(flat, [3] * len(flat), [10 < k < 18 for k in range(len(flat))]) == [3] * len(flat)  # straight: kept
    assert junction_turn_options(pts, [4] * len(pts), [k < 15 for k in range(len(pts))])[0] == 4  # run touches the start: kept
    assert junction_turn_options(pts[9:], [4] * (len(pts) - 9), junc[9:])[3] == 1  # only 2 m of road before the junction
