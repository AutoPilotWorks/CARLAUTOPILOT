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


def test_multicam_downscale_and_light_crop():
    import numpy as np

    env = pytest.importorskip("carlautopilot.carla_env.env")
    img = np.random.default_rng(0).integers(0, 256, (640, 1280, 3), dtype=np.uint8)
    small = env.box_downscale(img, 4)
    assert small.shape == (160, 320, 3) and small.dtype == np.uint8
    assert small[3, 5, 1] == round(img[12:16, 20:24, 1].mean())
    crop = env.light_crop(img, 120.0)
    # f = 369.5 px: 50 deg wide = 2 x 172 px; 18 deg up to 8 deg down = rows 200..372 (horizon at row 320)
    assert crop.shape == (172, 344, 3) and np.array_equal(crop[0, 0], img[200, 468])


def test_body_ahead_sees_a_tail_in_the_lane_when_the_centre_is_beside_it():
    r = WindowedRoute([(0, 0), (100, 0)])
    # 12 m bus turned 60 deg off the route: centre 4.3 m to the side, tail corner at 0.8 m
    bus = [(19.0, 0.8), (21.2, 2.0), (25.0, 9.2), (22.8, 8.0)]
    assert abs(r.project(22.0, 4.6)[1]) > 1.8  # the old centre test missed it
    assert r.body_ahead(bus, 5.0, 45.0, 1.3) == pytest.approx(19.0)
    assert r.body_ahead([(x, y + 3.5) for x, y in bus], 5.0, 45.0, 1.3) is None  # next lane over
    assert r.body_ahead([(x - 20.0, y) for x, y in bus], 5.0, 45.0, 1.3) is None  # behind us
    box = lambda cy, hw: [(20 + sx * 5.14, cy + sy * hw) for sx in (-1, 1) for sy in (-1, 1)]
    assert r.body_ahead(box(0.0, 1.97), 5.0, 45.0, 1.3) == pytest.approx(14.86)  # 3.9 m wide bus centred: every corner outside
    assert r.body_ahead(box(3.5, 1.97), 5.0, 45.0, 1.3) is None  # same bus one lane over
    # 4 m square turned 45 deg, nearest corner 2.5 m beside the route at s 17.2: it blocks where its edge comes within
    # 1.3 m, at s ~18.4 (the nearest corner said 17.2, the corner inside said 20.0)
    d = 4.0 / math.sqrt(2)
    diamond = [(20 - d, 2.5), (20, 2.5 + d), (20 + d, 2.5), (20, 2.5 - d)]
    assert r.body_ahead(diamond, 5.0, 45.0, 1.3) == pytest.approx(20 - d + 1.2, abs=0.2)
    assert r.body_ahead(diamond + diamond, 5.0, 45.0, 1.3) == pytest.approx(20 - d + 1.2, abs=0.2)  # CARLA gives 8 3-D corners


def test_radar_points_turn_into_sensor_frame_and_drop_road_returns():
    import numpy as np

    from carlautopilot.carla_env.route import radar_local, radar_obstacle_mask

    # [velocity, azimuth, altitude, depth]: 10 m dead ahead level, 10 m 30 deg right, a road return 6 deg down at 6 m
    pts = np.array([[-2.0, 0.0, 0.0, 10.0], [0.0, math.radians(30), 0.0, 10.0], [0.0, 0.0, math.radians(-6), 6.0]], dtype=np.float32)
    local = radar_local(pts)
    assert local[0] == pytest.approx([10.0, 0.0, 0.0], abs=1e-4)
    assert local[1] == pytest.approx([8.660, 5.0, 0.0], abs=1e-3)
    assert radar_obstacle_mask(local, mount_z=0.6).tolist() == [True, True, False]  # 0.6 - 0.63 m: the road


def test_radar_track_confirms_over_two_ticks_and_holds_through_short_gaps():
    from carlautopilot.carla_env.route import RadarTrack
    from carlautopilot.core import LeadInfo

    tr = RadarTrack(confirm=2, hold_s=0.3)
    d = LeadInfo(distance_m=10.0, rel_speed_mps=-2.0)
    assert tr.update(d, 5.0, 0.05) is None  # one tick: could be a ghost
    assert tr.update(d, 5.0, 0.05) == d
    held = [tr.update(None, 5.0, 0.05) for _ in range(7)]
    assert [round(h.distance_m, 2) for h in held[:6]] == [9.75, 9.5, 9.25, 9.0, 8.75, 8.5] and held[6] is None  # 0.3 s, then gone
    assert tr.update(d, 5.0, 0.05) is None  # lost: confirm again
    tr = RadarTrack(confirm=2, hold_s=0.3)
    assert tr.update(d, 5.0, 0.05) is None and tr.update(None, 5.0, 0.05) is None
    assert tr.update(d, 5.0, 0.05) == d  # 2 of the last 3 ticks: a sparse far return still confirms


def test_project_many_matches_project():
    import numpy as np

    r = WindowedRoute([(0, 0), (20, 0), (30, 10), (30, 40)])
    xy = np.random.default_rng(1).uniform(-5, 40, (50, 2))
    s, lat = r.project_many(xy, 5.0, 35.0)
    for (x, y), si, li in zip(xy, s, lat):
        s1, l1, _ = r.project(x, y, 5.0, 35.0)
        assert si == pytest.approx(s1) and li == pytest.approx(l1)


def test_points_seen_from_a_camera_shifted_left_lie_to_its_right():
    from carlautopilot.carla_env.route import points_seen_from

    assert points_seen_from([(10.0, 0.0)], 1.0, 0.0) == [pytest.approx((10.0, -1.0))]
    x, y = points_seen_from([(10.0, 0.0)], 0.0, 10.0)[0]  # camera yawed 10 deg left: the point is right of its axis
    assert y < 0 and x == pytest.approx(10.0 * math.cos(math.radians(10)))


def test_path_walkers_stand_on_the_lane_edge_or_cross_from_beyond_it():
    import random

    from carlautopilot.carla_env.route import path_walker_plan

    r = WindowedRoute([(0, 0), (300, 0)])
    plans = path_walker_plan(r, random.Random(3), 40)
    along = [p for p in plans if p["mode"] == "along"]
    cross = [p for p in plans if p["mode"] == "cross"]
    assert len(along) == len(cross) == 20
    assert all(40.0 <= p["s"] <= 270.0 for p in plans)
    assert all(-2.5 <= p["lat"] <= -2.0 and abs(math.sin(p["heading"])) < 1e-9 for p in along)  # right edge outside the 1.3 m corridor, along the road
    for p in cross:  # starts beyond the lane and walks toward the other side
        assert abs(p["lat"]) > 4.0 and math.sin(p["heading"]) * p["lat"] < 0 and 25.0 <= p["trigger_m"] <= 40.0


def test_lead_brake_cars_start_ahead_slower_than_the_ego_and_brake_in_separate_windows():
    import random

    from carlautopilot.carla_env.route import lead_brake_plan

    plans = lead_brake_plan(WindowedRoute([(0, 0), (300, 0)]), random.Random(5), 2)
    assert [round(p["s"] - 15.0 * k) >= 25 and p["s"] - 15.0 * k <= 45.0 for k, p in enumerate(plans)] == [True, True]
    for p in plans:
        assert 18.0 <= p["speed_kmh"] <= 26.0  # below the expert's 8 m/s (28.8 km/h), so the ego closes in
        ends = [t + d for t, d in p["brakes"]]
        assert p["brakes"][0][0] >= 10.0 and all(1.5 <= d <= 5.0 for _, d in p["brakes"])
        assert all(nxt >= end + 10.0 for end, (nxt, _) in zip(ends, p["brakes"][1:]))  # >= 10 s of driving between stops
