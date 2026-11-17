from __future__ import annotations

import math
import queue
import random
import subprocess
import sys
import time
from pathlib import Path

import carla
import numpy as np

from carlautopilot.carla_env.route import (
    LANE_CHANGE_OPTIONS,
    WindowedRoute,
    carla_to_xy,
    carla_yaw_to_rad,
    commands_from_options,
    dedupe_points,
    junction_turn_options,
    lead_brake_plan,
    path_walker_plan,
    RadarTrack,
    radar_local,
    radar_obstacle_mask,
)
from carlautopilot.core import Action, Command, Event, EventKind, GroundTruth, LeadInfo, Observation, StepResult
from carlautopilot.scenario import RouteCommand, ScenarioSpec, Waypoint
from carlautopilot.sim.fake_env import (
    BLOCKED_SPEED_MPS,
    LEAD_VISIBLE_M,
    LOOKAHEAD_OFFSETS_M,
    OFF_ROAD_LATERAL_M,
    OFF_ROAD_SECONDS,
    RED_LIGHT_SPEED_MPS,
    STOP_SIGN_HOLD_S,
    STOP_SIGN_ZONE_M,
    _wrap,
    route_target,
)

VEHICLE = "vehicle.lincoln.mkz_2020"
WALKER_CROSS_FACTOR = 0.2
YIELD_LOOK_M = 30.0  # yield is checked while the junction entry is at most this far ahead
YIELD_HORIZON_S, YIELD_STEP_S = 4.0, 0.25
YIELD_MIN_EGO_MPS = 3.0  # a stopped ego is assumed to pull away at this speed when predicting conflicts
YIELD_CLEAR_M = {"vehicle": 2.8, "walker": 2.0}
WHEELBASE_M = 2.86  # measured 2026-10-21
MAX_STEER_RAD = math.radians(70.0)
FRONT_CAMERA = (1.5, 0.0, 1.6)  # x, y, z m in the vehicle frame
# sedan camera rig (stage 1 probe 2026-10-31, D:\carlautopilot_data\rig_probe): the front camera renders at
# MULTICAM_SCALE x the image size; the driving image is its box-downscale and the traffic-light image a native-res crop,
# so one physical front camera replaces the truck's separate light camera. At 120 deg / 1280 px a light head at 40 m is
# ~3.5x11 px (the truck's 50 deg / 320 px light camera: 3.3x10). Surround cameras at the mirrors and above the rear plate, name ->
# (x, y, z, yaw, pitch, fov) in the vehicle frame (CARLA, y right): 150 deg at the mirrors halves the ground blind area
# within 2 m of the body vs 120 deg (8.1 vs 16.8 m2 at pitch -20/-25).
MULTICAM_SCALE = 4
VIEW_SCALE = 2  # recovery views need no light crop: 2x + box-downscale renders 1/4 of 4x's pixels (4x: collection 2.6x slower, 11-02)
# front radars in the bumper like a production ACC/AEB set: a narrow long-range one and a wide short-range one (the
# 30 deg radar alone covered only +-0.5 m at 2 m, so walkers close ahead went unseen, 11-02 probe). x, z m in the
# vehicle frame, horizontal/vertical fov deg, range m, points per second
RADARS = ((2.3, 0.6, 30.0, 10.0, 50.0, 6000), (2.4, 0.5, 90.0, 20.0, 30.0, 6000))
RADAR_CONFIRM_TICKS = 2  # an obstacle counts once seen in this many ticks in a row (single-tick ghosts are dropped)
RADAR_HOLD_S = 0.3  # a confirmed obstacle is kept this long through missed ticks, its gap shrinking with our speed
LIGHT_CROP_DEG = (50.0, 18.0, 8.0)  # width, above and below the horizon: the truck light camera's view (50 deg, up 5)
SURROUND_CAMERAS = {
    "left": (0.6, -1.05, 1.1, -90.0, -20.0, 150.0),
    "right": (0.6, 1.05, 1.1, 90.0, -20.0, 150.0),
    "rear": (-2.3, 0.0, 1.05, 180.0, -15.0, 120.0),
}


def box_downscale(img: np.ndarray, k: int) -> np.ndarray:
    """(H*k, W*k, 3) uint8 -> (H, W, 3): rounded mean of each k x k block (anti-aliased, deterministic)."""
    h, w = img.shape[0] // k, img.shape[1] // k
    s = img[: h * k, : w * k].reshape(h, k, w, k, 3).sum(axis=(1, 3), dtype=np.uint32)
    return ((s + k * k // 2) // (k * k)).astype(np.uint8)


def light_crop(img: np.ndarray, fov_deg: float, crop: tuple[float, float, float] = LIGHT_CROP_DEG) -> np.ndarray:
    """Centre crop of a pinhole image: crop[0] deg wide, from crop[1] deg above the horizon to crop[2] deg below."""
    h, w = img.shape[:2]
    f = w / (2 * math.tan(math.radians(fov_deg) / 2))
    half = round(f * math.tan(math.radians(crop[0] / 2)))
    top, bottom = round(h / 2 - f * math.tan(math.radians(crop[1]))), round(h / 2 + f * math.tan(math.radians(crop[2])))
    return np.ascontiguousarray(img[max(top, 0):bottom, w // 2 - half:w // 2 + half])
ROUTE_END_MARGIN_M = 2.0
BLOCKED_SECONDS = 90.0  # CARLA reds at 4-way junctions can exceed 30 s
CORRIDOR_HALF_M = 1.8
LEAD_BODY_HALF_M = 1.3  # a body outline this close to the route centre line is in our path (sedan half width ~1.0)
# pedestrians get a wider corridor: one stopped 1.4-1.6 m beside the centre line (outline ~1.1-1.3 m) was passed at
# 18 km/h and hit by the mirror side (Town03_Opt_100_000, 11-17). 1.8 m also caught the scripted walkers along the lane
# edge and the ego trailed them; 1.5 m leaves ~0.45 m between the mirror and a passed walker.
WALKER_BODY_HALF_M = 1.5
# scripted walkers stop when their next WALKER_HALT_LOOK_M of path enters the ego footprint grown by
# WALKER_HALT_MARGIN_M: one whose line passed 1.7 m from the ego centre walked into the standing car's front corner,
# because the old rule treated the ego as its centre point (Town03_Opt_100_003, 11-17)
WALKER_HALT_LOOK_M = 3.0
WALKER_HALT_MARGIN_M = 0.5
# a walker held that long turns back, like a pedestrian stepping back for a car: held walkers and an ego waiting for
# them deadlocked (Town03 radar probe 3 of 4 blocked, 11-17)
WALKER_TURN_BACK_S = 4.0
WALKER_STUCK_S = 10.0  # a scripted walker that has not moved this long while free to walk is removed
LIGHT_MATCH_M = 2.0
SETTLE_TICKS = 20
WEATHERS = ("ClearNoon", "CloudyNoon", "WetNoon", "ClearSunset", "CloudySunset")


_SWITCH = (
    "import sys, carla\n"
    "c = carla.Client(sys.argv[1], int(sys.argv[2])); c.set_timeout(180.0)\n"
    "if not c.get_world().get_map().name.endswith(sys.argv[3]):\n"
    "    c.load_world(sys.argv[3])\n"
)


def switch_map(town: str, host: str = "localhost", port: int = 2000, tries: int = 3) -> None:
    """Loads the town from a separate process: load_world intermittently aborts the client
    process (assert '_episode != nullptr' in Simulator.h, seen with 0.9.16 on Windows)."""
    for _ in range(tries):
        if subprocess.run([sys.executable, "-c", _SWITCH, host, str(port), town]).returncode == 0:
            return
        time.sleep(5.0)
    raise RuntimeError(f"could not load {town}")


def _load_agents(carla_root: str | Path) -> None:
    p = str(Path(carla_root) / "PythonAPI" / "carla")
    if p not in sys.path:
        sys.path.append(p)


class CarlaEnv:
    """Env adapter for a running CARLA 0.9.16 server (synchronous mode, fixed dt).

    CARLA's steer sign already matches the project (positive = right); positions are converted
    to the project frame (y flipped). Scenario meta: town, start [x, y, z, yaw] in CARLA
    coordinates, npc_vehicles, weather.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = 8000,
        dt: float = 0.05,
        image_size: tuple[int, int] = (128, 256),
        render_images: bool = True,
        privileged_obs: bool = False,
        timeout_s: float = 60.0,
        follow_spectator: bool = True,
        fov: float = 120.0,
        junction_rules: bool = False,
        radar: bool = False,
        extra_views: tuple[tuple[float, float], ...] = (),
        walker_cross: float = WALKER_CROSS_FACTOR,
        path_walkers: int = 0,
        lead_brakes: int = 0,
    ):
        """The sedan camera rig (see SURROUND_CAMERAS): one fixed front camera (fov deg) rendered at MULTICAM_SCALE x
        image_size gives obs.image (downscaled) and extras["light_image"] (light_crop), and extras["surround_images"]
        holds {left, right, rear} at image_size.

        junction_rules: stop signs on the route must be served (halt in STOP_SIGN_ZONE_M for STOP_SIGN_HOLD_S, else a
        STOP_SIGN event), and privileged extras gain next_stop_sign_distance_m and yield_distance_m (junction entry ahead
        while a vehicle/pedestrian is predicted, at constant velocity, to cross the ego's path inside the junction).

        radar: front RADAR sensors (RADARS); extras["radar"] holds their raw detections and extras["radar_lead"] the
        nearest detection above the road and inside the route corridor, confirmed over RADAR_CONFIRM_TICKS and held
        RADAR_HOLD_S (rules: the route is the planner's, not perception).

        extra_views: recovery cameras, (offset left m, yaw left deg) from the front camera, rendered at VIEW_SCALE x and
        box-downscaled into privileged extras["view_images"] for collection; the expert labels them from that pose.
        walker_cross: share of pedestrians that cross roads (CARLA's cross factor).
        path_walkers: scripted pedestrians per episode in the ego's path (route.path_walker_plan), seeded by the scenario.
        lead_brakes: vehicles put ahead on the ego's route that brake hard now and then (route.lead_brake_plan).
        """
        self.path_walkers = path_walkers
        self._path_walkers: list[list] = []  # [actor, plan, started, held since, last moved t, last moved location]
        self.lead_brakes = lead_brakes
        self._lead_brakes: list[list] = []  # [actor, plan, braking]
        self.extra_views = tuple(extra_views)
        self._view_queues: list[queue.Queue] = []
        self._view_images: list[np.ndarray] = []
        self.walker_cross = walker_cross
        self.radar = radar
        self._radars: list[tuple] = []  # (actor, queue, mount z)
        self._radar_points: list[np.ndarray] = []
        self._radar_track = RadarTrack(RADAR_CONFIRM_TICKS, RADAR_HOLD_S)
        self.radar_ms = 0.0
        self.lead_type: str | None = None
        self.lead_id: int | None = None  # actor id of the simulator's lead (radar probes count catches per object)
        self._surround_queues: dict[str, queue.Queue] = {}
        self._surround_images: dict[str, np.ndarray] = {}
        self._light_image: np.ndarray | None = None
        self.junction_rules = junction_rules
        self.fov = fov
        self.dt = dt
        self.follow_spectator = follow_spectator
        self.image_size = image_size
        self.render_images = render_images
        self.privileged_obs = privileged_obs
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout_s)
        self.world = self.client.get_world()
        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm_port = tm_port
        self._original_settings = self.world.get_settings()
        self._actors: list = []
        self._npcs: list[int] = []
        self._ego = None
        self._scenario: ScenarioSpec | None = None
        self._front_queue: queue.Queue | None = None
        self._last_image: np.ndarray | None = None

    def check_town(self, town: str) -> None:
        name = self.world.get_map().name
        if not name.endswith(town):
            raise RuntimeError(f"server has {name}, scenario needs {town}; call switch_map() before creating the env")

    def _sync(self, on: bool) -> None:
        s = self.world.get_settings()
        s.synchronous_mode = on
        s.fixed_delta_seconds = self.dt if on else None
        self.world.apply_settings(s)
        self.tm.set_synchronous_mode(on)

    def make_route_scenario(
        self,
        name: str,
        rng: random.Random,
        carla_root: str | Path,
        min_len_m: float = 250.0,
        max_len_m: float = 600.0,
        npc_vehicles: int = 30,
        speed_limit_mps: float = 8.0,
        npc_walkers: int = 0,
        max_tries: int = 200,
    ) -> ScenarioSpec:
        """Random A->B route from CARLA's GlobalRoutePlanner, with junction commands."""
        _load_agents(carla_root)
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        cmap = self.world.get_map()
        grp = GlobalRoutePlanner(cmap, 2.0)
        spawns = cmap.get_spawn_points()
        for _ in range(max_tries):
            a, b = rng.sample(spawns, 2)
            trace = grp.trace_route(a.location, b.location)
            if any(int(opt) in LANE_CHANGE_OPTIONS for _, opt in trace):
                continue
            pts = [carla_to_xy(wp.transform.location.x, wp.transform.location.y) for wp, _ in trace]
            keep = dedupe_points(pts)
            pts = [pts[i] for i in keep]
            opts = junction_turn_options(pts, [int(trace[i][1]) for i in keep], [bool(trace[i][0].is_junction) for i in keep])
            if len(pts) < 2:
                continue
            s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(np.array(pts), axis=0).T))])
            if not min_len_m <= s[-1] <= max_len_m:
                continue
            start = trace[0][0].transform
            return ScenarioSpec(
                name=name,
                route=tuple(Waypoint(x, y) for x, y in pts),
                commands=commands_from_options(list(s), opts),
                max_time_s=float(s[-1] / speed_limit_mps * 3.0 + 120.0),
                speed_limit_mps=speed_limit_mps,
                seed=rng.randrange(1 << 30),
                tags=("carla",),
                meta={
                    "town": cmap.name.split("/")[-1],
                    "start": [start.location.x, start.location.y, start.location.z, start.rotation.yaw],
                    "npc_vehicles": npc_vehicles,
                    "npc_walkers": npc_walkers,
                    "weather": rng.choice(WEATHERS),
                },
            )
        raise RuntimeError("could not find a route in the length range")

    def reset(self, scenario: ScenarioSpec, seed: int | None = None) -> Observation:
        self._cleanup()
        scenario.validate()
        meta = scenario.meta
        self.check_town(meta["town"])
        self._sync(True)
        seed = scenario.seed if seed is None else seed
        rng = random.Random(seed)
        self.tm.set_random_device_seed(seed)
        self.world.set_weather(getattr(carla.WeatherParameters, meta.get("weather", "ClearNoon")))
        bpl = self.world.get_blueprint_library()

        x, y, z, yaw = meta["start"]
        start = carla.Transform(carla.Location(x=x, y=y, z=z + 0.5), carla.Rotation(yaw=yaw))
        self._ego = self.world.spawn_actor(bpl.find(VEHICLE), start)
        self._actors.append(self._ego)

        self._front_queue, self._surround_queues = None, {}
        if self.render_images:
            h, w = self.image_size
            x, y, z = FRONT_CAMERA
            self._front_queue = self._camera(bpl, w * MULTICAM_SCALE, h * MULTICAM_SCALE, self.fov, carla.Transform(carla.Location(x=x, y=y, z=z)))
            for name, (x, y, z, yaw, pitch, fov) in SURROUND_CAMERAS.items():
                self._surround_queues[name] = self._camera(bpl, w, h, fov, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw)))
            x, y, z = FRONT_CAMERA
            # CARLA y is right and yaw clockwise, the views are given left-positive
            self._view_queues = [self._camera(bpl, w * VIEW_SCALE, h * VIEW_SCALE, self.fov, carla.Transform(carla.Location(x=x, y=y - dy, z=z), carla.Rotation(yaw=-dyaw)))
                                 for dy, dyaw in self.extra_views]
        self._radars = []
        self._radar_track = RadarTrack(RADAR_CONFIRM_TICKS, RADAR_HOLD_S)
        for x, z, hfov, vfov, rng_m, pps in RADARS if self.radar else ():
            bp = bpl.find("sensor.other.radar")
            for k, v in (("horizontal_fov", hfov), ("vertical_fov", vfov), ("range", rng_m), ("points_per_second", pps)):
                bp.set_attribute(k, str(v))
            actor = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=x, z=z)), attach_to=self._ego)
            q = queue.Queue()
            actor.listen(q.put)
            self._actors.append(actor)
            self._radars.append((actor, q, z))
        self._collisions: list = []
        col = self.world.spawn_actor(bpl.find("sensor.other.collision"), carla.Transform(), attach_to=self._ego)
        col.listen(self._collisions.append)
        self._actors.append(col)

        self._spawn_npcs(rng, int(meta.get("npc_vehicles", 0)), start.location)
        self._spawn_walkers(seed, int(meta.get("npc_walkers", 0)), start.location)

        self._scenario = scenario
        self._route = WindowedRoute([(w.x, w.y) for w in scenario.route])
        self._spawn_path_walkers(random.Random(seed + 2))
        self._spawn_lead_brakes(random.Random(seed + 3))
        self._lights = self._lights_on_route()
        self._stop_signs = self._stop_signs_on_route() if self.junction_rules else []
        self._stop_served: set[int] = set()
        self._stop_hold = 0.0
        self._junctions = self._junctions_on_route() if self.junction_rules else []
        self._light_events_done: set[int] = set()
        self._s = 0.0
        self.t = 0.0
        self._off_road_time = 0.0
        self._blocked_time = 0.0
        self._done = False
        for _ in range(SETTLE_TICKS):
            self._ego.apply_control(carla.VehicleControl(brake=1.0))
            self._tick()
        self._collisions.clear()
        self._update_pose()
        return self._observe()

    def _spawn_npcs(self, rng: random.Random, n: int, ego_loc) -> None:
        if n <= 0:
            return
        bps = [b for b in self.world.get_blueprint_library().filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        spawns = [sp for sp in self.world.get_map().get_spawn_points() if sp.location.distance(ego_loc) > 15.0]
        rng.shuffle(spawns)
        for sp in spawns[:n]:
            npc = self.world.try_spawn_actor(rng.choice(bps), sp)
            if npc is not None:
                npc.set_autopilot(True, self.tm_port)
                self._npcs.append(npc.id)

    def _spawn_path_walkers(self, rng: random.Random) -> None:
        self._path_walkers = []
        if self.path_walkers <= 0:
            return
        bps = list(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
        cmap = self.world.get_map()
        for plan in path_walker_plan(self._route, rng, self.path_walkers):
            px, py = self._route.point_at(plan["s"])
            h = self._route.heading_at(plan["s"])
            px, py = px - math.sin(h) * plan["lat"], py + math.cos(h) * plan["lat"]
            cx, cy = px, -py  # project -> CARLA
            z = cmap.get_waypoint(carla.Location(x=cx, y=cy), project_to_road=True).transform.location.z
            bp = rng.choice(bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            w = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(x=cx, y=cy, z=z + 1.0)))
            if w is not None:
                self._actors.append(w)
                self._path_walkers.append([w, plan, plan["mode"] == "along", None, None, None])

    def _spawn_lead_brakes(self, rng: random.Random) -> None:
        """Traffic Manager cars on the ego's own route ahead (set_path, no lane changes, a slower desired speed), so the
        ego catches up and follows them; _drive_lead_brakes makes them brake to a stop in their planned windows."""
        self._lead_brakes = []
        if self.lead_brakes <= 0:
            return
        bps = [b for b in self.world.get_blueprint_library().filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        cmap = self.world.get_map()
        for plan in lead_brake_plan(self._route, rng, self.lead_brakes):
            px, py = self._route.point_at(plan["s"])
            h = self._route.heading_at(plan["s"])
            wp = cmap.get_waypoint(carla.Location(x=px, y=-py), project_to_road=True)
            tr = carla.Transform(wp.transform.location + carla.Location(z=0.5), carla.Rotation(yaw=-math.degrees(h)))
            v = self.world.try_spawn_actor(rng.choice(bps), tr)
            if v is None:
                continue
            self._npcs.append(v.id)  # batch-destroyed like the NPCs: destroy() on a Traffic Manager car crashed the client (11-02 smoke)
            v.set_autopilot(True, self.tm_port)
            self.tm.auto_lane_change(v, False)
            self.tm.set_desired_speed(v, plan["speed_kmh"])
            path = [carla.Location(x=x, y=-y) for x, y in (self._route.point_at(s) for s in np.arange(plan["s"] + 5.0, self._route.length, 5.0))]
            if path:
                self.tm.set_path(v, path)
            self._lead_brakes.append([v, plan, False])

    def _drive_lead_brakes(self) -> None:
        """The Traffic Manager keeps driving the car and only its desired speed drops to 0 in a brake window. Switching
        autopilot off to brake did not hold: the car sped up to 50-60 km/h inside the window and, re-registered, lost
        its desired speed and path; Town05_Opt_211_007 then aborted in world.tick (11-02, 11-17)."""
        for item in self._lead_brakes:
            v, plan, braking = item
            now = any(t0 <= self.t < t0 + d for t0, d in plan["brakes"])
            if now != braking:
                self.tm.set_desired_speed(v, 0.0 if now else plan["speed_kmh"])
                item[2] = now

    def _drive_path_walkers(self) -> None:
        """Scripted walkers keep their line but halt while their next few metres would enter the ego's footprint, so they
        never walk into a standing car (11-01 smoke). Only their own path counts: a walker halting for a car merely
        beside its line (e.g. mid-crossing in front of it) deadlocked with the ego waiting for it (11-02 probe).
        Walkers along the road follow its bends at their planned lateral offset: walking straight from their start they
        drifted into the lane on curves. One that has not moved for WALKER_STUCK_S without being held (caught on
        something) is removed; one stood in the lane and the ego waited behind it to the end (Town02_Opt_100_001, 11-18)."""
        tr = self._ego.get_transform()
        f, r = tr.get_forward_vector(), tr.get_right_vector()
        ext = self._ego.bounding_box.extent
        hx, hy = ext.x + WALKER_HALT_MARGIN_M, ext.y + WALKER_HALT_MARGIN_M
        gone = []
        for item in self._path_walkers:
            w, plan, started, held_since = item[:4]
            if not started and plan["s"] - self._s <= plan["trigger_m"]:
                item[2] = started = True
            if started:
                loc = w.get_location()
                hp = plan["heading"]
                if plan["mode"] == "along":
                    s, lat, _ = self._route.project(*carla_to_xy(loc.x, loc.y), plan.setdefault("s_now", plan["s"]) - 15.0, plan["s_now"] + 15.0)
                    plan["s_now"] = s
                    h = self._route.heading_at(s)
                    sign = 1.0 if math.cos(plan["heading"] - h) >= 0.0 else -1.0
                    pull = 0.5 * (plan["lat"] - lat)  # back toward the planned offset (left normal: -sin h, cos h)
                    hp = math.atan2(sign * math.sin(h) + pull * math.cos(h), sign * math.cos(h) - pull * math.sin(h))
                dx, dy = math.cos(hp), -math.sin(hp)  # CARLA frame
                blocked = False
                for d in np.arange(0.0, WALKER_HALT_LOOK_M + 1e-6, 0.5):
                    px, py = loc.x + dx * d - tr.location.x, loc.y + dy * d - tr.location.y
                    if abs(px * f.x + py * f.y) < hx and abs(px * r.x + py * r.y) < hy:
                        blocked = True
                        break
                if blocked or item[5] is None or loc.distance(item[5]) > 0.5:
                    item[4], item[5] = self.t, loc  # moving, or held on purpose
                elif self.t - item[4] >= WALKER_STUCK_S:
                    gone.append(item)
                    continue
                if not blocked:
                    item[3] = None
                elif held_since is None:
                    item[3] = self.t
                elif self.t - held_since >= WALKER_TURN_BACK_S:
                    plan["heading"] += math.pi
                    item[3] = None
                    continue
                w.apply_control(carla.WalkerControl(direction=carla.Vector3D(dx, dy, 0.0), speed=0.0 if blocked else plan["speed"]))
        for item in gone:
            self._path_walkers.remove(item)
            self._actors.remove(item[0])
            item[0].destroy()

    def _spawn_walkers(self, seed: int, n: int, ego_loc) -> None:
        if n <= 0:
            return
        bpl = self.world.get_blueprint_library()
        rng = random.Random(seed + 1)
        self.world.set_pedestrians_seed(seed)
        self.world.set_pedestrians_cross_factor(self.walker_cross)
        walkers = []
        for _ in range(n * 2):
            if len(walkers) >= n:
                break
            loc = self.world.get_random_location_from_navigation()
            if loc is None or loc.distance(ego_loc) < 10.0:
                continue
            bp = rng.choice(bpl.filter("walker.pedestrian.*"))
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            w = self.world.try_spawn_actor(bp, carla.Transform(loc + carla.Location(z=1.0)))
            if w is not None:
                walkers.append(w)
                self._actors.append(w)
        self.world.tick()
        ctrl_bp = bpl.find("controller.ai.walker")
        for w in walkers:
            c = self.world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            self._actors.append(c)
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            c.set_max_speed(1.0 + rng.random())

    def _lights_on_route(self) -> list[tuple[float, object]]:
        found = []
        for light in self.world.get_actors().filter("traffic.traffic_light"):
            for wp in light.get_stop_waypoints():
                lx, ly = carla_to_xy(wp.transform.location.x, wp.transform.location.y)
                s, _, dist = self._route.project(lx, ly)
                heading = carla_yaw_to_rad(wp.transform.rotation.yaw)
                if dist < LIGHT_MATCH_M and abs(_wrap(heading - self._route.heading_at(s))) < math.radians(30):
                    found.append((s, light))
        return sorted(found, key=lambda f: f[0])

    def _stop_signs_on_route(self) -> list[float]:
        cmap = self.world.get_map()
        found = []
        for sign in self.world.get_actors().filter("traffic.stop"):
            loc = sign.get_transform().transform(sign.trigger_volume.location)
            wp = cmap.get_waypoint(loc)
            lx, ly = carla_to_xy(wp.transform.location.x, wp.transform.location.y)
            s, _, dist = self._route.project(lx, ly)
            heading = carla_yaw_to_rad(wp.transform.rotation.yaw)
            if dist < LIGHT_MATCH_M and abs(_wrap(heading - self._route.heading_at(s))) < math.radians(30):
                found.append(s)
        return sorted(found)

    def _junctions_on_route(self) -> list[tuple[float, float]]:
        cmap = self.world.get_map()
        spans, start = [], None
        n = int(self._route.length // 2.0) + 1
        for i in range(n + 1):
            s = min(i * 2.0, self._route.length)
            x, y = self._route.point_at(s)
            inside = cmap.get_waypoint(carla.Location(x=x, y=-y, z=0.0)).is_junction
            if inside and start is None:
                start = s
            elif not inside and start is not None:
                spans.append((start, s))
                start = None
        if start is not None:
            spans.append((start, self._route.length))
        return spans

    def _next_stop_sign(self) -> float | None:
        for i, ss in enumerate(self._stop_signs):
            if i not in self._stop_served and ss - self._s > -1.0:
                return ss - self._s
        return None

    def _serve_stop_signs(self) -> list[Event]:
        events = []
        for i, ss in enumerate(self._stop_signs):
            if i in self._stop_served:
                continue
            d = ss - self._s
            if d < -1.0:
                self._stop_served.add(i)
                events.append(Event(self.t, EventKind.STOP_SIGN, f"stop sign at s={ss:.0f}"))
            elif d <= STOP_SIGN_ZONE_M and self.v < BLOCKED_SPEED_MPS:
                self._stop_hold += self.dt
                if self._stop_hold >= STOP_SIGN_HOLD_S:
                    self._stop_served.add(i)
                    self._stop_hold = 0.0
            break
        return events

    def _yield_distance(self) -> float | None:
        entry = next((a for a, b in self._junctions if a - self._s > -0.5), None)
        if entry is None or entry - self._s > YIELD_LOOK_M:
            return None
        v = max(self.v, YIELD_MIN_EGO_MPS)
        steps = [k * YIELD_STEP_S for k in range(int(YIELD_HORIZON_S / YIELD_STEP_S) + 1)]
        ego = [(t, self._route.point_at(self._s + v * t)) for t in steps if self._s + v * t >= entry - 2.0]
        if not ego:
            return None
        reach = v * YIELD_HORIZON_S + 40.0
        for kind in ("vehicle", "walker"):
            for a in self.world.get_actors().filter(f"{kind}.*"):
                if a.id == self._ego.id:
                    continue
                loc = a.get_location()
                ox, oy = carla_to_xy(loc.x, loc.y)
                if math.hypot(ox - self.x, oy - self.y) > reach:
                    continue
                vel = a.get_velocity()
                vx, vy = carla_to_xy(vel.x, vel.y)
                if kind == "vehicle":
                    s, lat, _ = self._route.project(ox, oy, self._s, self._s + reach)
                    if abs(lat) < CORRIDOR_HALF_M and math.cos(math.atan2(vy, vx) - self._route.heading_at(s)) > 0.87 and math.hypot(vx, vy) > 0.5:
                        continue  # same lane, same direction: the ACC lead, not a crossing conflict
                clear = YIELD_CLEAR_M[kind]
                hit = next((t for t, (ex, ey) in ego if math.hypot(ox + vx * t - ex, oy + vy * t - ey) < clear), None)
                if hit is not None:
                    self.yield_reason = (a.type_id, round(math.hypot(vx, vy), 1), round(math.hypot(ox - self.x, oy - self.y), 1), hit)
                    return entry - self._s
        return None

    def _camera(self, bpl, width: int, height: int, fov: float, mount) -> queue.Queue:
        bp = bpl.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        cam = self.world.spawn_actor(bp, mount, attach_to=self._ego)
        q: queue.Queue = queue.Queue()
        cam.listen(q.put)
        self._actors.append(cam)
        return q

    @staticmethod
    def _wait(q: queue.Queue, frame: int):
        while True:
            m = q.get(timeout=10.0)
            if m.frame >= frame:
                return m

    @classmethod
    def _read(cls, q: queue.Queue, frame: int) -> np.ndarray:
        img = cls._wait(q, frame)
        a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        return np.ascontiguousarray(a[:, :, 2::-1])

    def _tick(self) -> None:
        frame = self.world.tick()
        if self._front_queue is not None:
            front = self._read(self._front_queue, frame)
            self._light_image = light_crop(front, self.fov)
            self._last_image = box_downscale(front, MULTICAM_SCALE)
        self._surround_images = {n: self._read(q, frame) for n, q in self._surround_queues.items()}
        self._view_images = [box_downscale(self._read(q, frame), VIEW_SCALE) for q in self._view_queues]
        self._radar_points = [np.frombuffer(self._wait(q, frame).raw_data, dtype=np.float32).reshape(-1, 4).copy() for _, q, _ in self._radars]

    def _radar_lead(self) -> LeadInfo | None:
        """Nearest radar detection above the road inside the route corridor ahead, over all radars; closing speed from
        its Doppler velocity (radial only: a pedestrian crossing straight across reads ~0)."""
        t0 = time.perf_counter()
        best = None
        ego_half = self._ego.bounding_box.extent.x
        for (actor, _, mount_z), pts in zip(self._radars, self._radar_points):
            local = radar_local(pts)
            keep = radar_obstacle_mask(local, mount_z)
            if not keep.any():
                continue
            m = np.array(actor.get_transform().get_matrix())
            world = (m[:3, :3] @ local[keep].T).T + m[:3, 3]
            s, lat = self._route.project_many(np.stack([world[:, 0], -world[:, 1]], axis=1), self._s, self._s + LEAD_VISIBLE_M)  # CARLA -> project frame
            # the vehicle corridor: widening it to the pedestrians' 1.5 m barely helped walkers (Town03 only) and let
            # nearer roadside returns hide the lead car (Town03 vehicle frames 71% -> 42%, 11-18)
            inside = np.nonzero((s > self._s) & (np.abs(lat) < LEAD_BODY_HALF_M))[0]
            if len(inside):
                k = inside[np.argmin(s[inside])]
                gap = max(float(s[k]) - self._s - ego_half, 0.0)
                if best is None or gap < best.distance_m:
                    best = LeadInfo(distance_m=gap, rel_speed_mps=float(pts[keep, 0][k]))
        self.radar_ms = (time.perf_counter() - t0) * 1000.0
        return self._radar_track.update(best, self.v, self.dt)

    def _follow(self, tr) -> None:
        back = tr.get_forward_vector()
        loc = tr.location - carla.Location(x=back.x * 8.0, y=back.y * 8.0) + carla.Location(z=4.0)
        self.world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-15.0, yaw=tr.rotation.yaw)))

    def _update_pose(self) -> None:
        tr = self._ego.get_transform()
        if self.follow_spectator:
            self._follow(tr)
        vel = self._ego.get_velocity()
        self.x, self.y = carla_to_xy(tr.location.x, tr.location.y)
        self.yaw = carla_yaw_to_rad(tr.rotation.yaw)
        self.v = math.hypot(vel.x, vel.y)
        self._s, self._lat, _ = self._route.project(self.x, self.y, self._s - 5.0, self._s + 20.0)

    def _command(self) -> Command:
        cmd = Command.FOLLOW
        for rc in self._scenario.commands:
            if rc.s_m <= self._s:
                cmd = rc.command
        return cmd

    def _lead_info(self) -> LeadInfo | None:
        ego_half = self._ego.bounding_box.extent.x
        best: LeadInfo | None = None
        others = list(self.world.get_actors().filter("vehicle.*")) + list(self.world.get_actors().filter("walker.pedestrian.*"))
        for a in others:
            if a.id == self._ego.id:
                continue
            loc = a.get_location()
            ox, oy = carla_to_xy(loc.x, loc.y)
            if math.hypot(ox - self.x, oy - self.y) > LEAD_VISIBLE_M + 10.0:
                continue
            corners = [carla_to_xy(v.x, v.y) for v in a.bounding_box.get_world_vertices(a.get_transform())]
            half = WALKER_BODY_HALF_M if a.type_id.startswith("walker") else LEAD_BODY_HALF_M
            s = self._route.body_ahead(corners, self._s, self._s + LEAD_VISIBLE_M, half)
            if s is None:
                continue
            gap = s - self._s - ego_half
            if gap > LEAD_VISIBLE_M:
                continue
            h = self._route.heading_at(s)
            vel = a.get_velocity()
            vx, vy = carla_to_xy(vel.x, vel.y)
            along = vx * math.cos(h) + vy * math.sin(h)
            if best is None or gap < best.distance_m:
                best = LeadInfo(distance_m=max(gap, 0.0), rel_speed_mps=along - self.v)
                self.lead_type, self.lead_id = a.type_id, a.id
        if best is None:
            self.lead_type = self.lead_id = None
        return best

    def _extras(self) -> dict:
        if not self.privileged_obs:
            return {}
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        pts = []
        for d in LOOKAHEAD_OFFSETS_M:
            px, py = self._route.point_at(self._s + d)
            dx, dy = px - self.x, py - self.y
            pts.append((dx * c - dy * s, dx * s + dy * c))
        extras = {
            "heading_error_rad": _wrap(self.yaw - self._route.heading_at(self._s)),
            "lateral_error_m": self._lat,
            "lookahead_points": pts,
            "view_images": self._view_images,
        }
        for ls, light in self._lights:
            dist = ls - self._s
            if 0.0 < dist <= LEAD_VISIBLE_M:
                extras["next_light_distance_m"] = dist
                extras["next_light_red"] = light.get_state() in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)
                break
        if self.junction_rules:
            d = self._next_stop_sign()
            if d is not None and d <= LEAD_VISIBLE_M:
                extras["next_stop_sign_distance_m"] = max(d, 0.0)
            # every sign ahead, served or not: what the camera sees (the served state is the driver's memory)
            d = next((ss - self._s for ss in self._stop_signs if ss - self._s > -1.0), None)
            if d is not None and d <= LEAD_VISIBLE_M:
                extras["stop_sign_distance_m"] = max(d, 0.0)
            y = self._yield_distance()
            if y is not None:
                extras["yield_distance_m"] = max(y, 0.0)
        return extras

    def _observe(self) -> Observation:
        image = self._last_image if self.render_images else None
        extras = self._extras()
        extras["route_target"] = route_target(self._route, self._s, self.x, self.y, self.yaw)
        if self.render_images:
            extras["light_image"] = self._light_image
            extras["surround_images"] = self._surround_images
        if self.radar:
            extras["radar"] = np.concatenate(self._radar_points) if self._radar_points else np.zeros((0, 4), dtype=np.float32)
            extras["radar_lead"] = self._radar_lead()
        return Observation(t=self.t, speed_mps=self.v, command=self._command(), image=image, lead=self._lead_info(), extras=extras)

    def ground_truth(self) -> GroundTruth:
        nxt = next(((ls - self._s, light) for ls, light in self._lights if 0.0 < ls - self._s <= LEAD_VISIBLE_M), None)
        red = None if nxt is None else nxt[1].get_state() in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)
        return GroundTruth(self.x, self.y, self.yaw, self.v, self._s, self._route.length, self._lat, None if nxt is None else nxt[0], red)

    def step(self, action: Action) -> StepResult:
        if self._scenario is None:
            raise RuntimeError("call reset() first")
        if self._done:
            raise RuntimeError("episode finished; call reset()")
        a = action.clipped()
        self._ego.apply_control(carla.VehicleControl(throttle=a.throttle, steer=a.steer, brake=a.brake))
        prev_s = self._s
        self._drive_path_walkers()
        self._drive_lead_brakes()
        self._tick()
        self.t += self.dt
        self._update_pose()

        events: list[Event] = []
        termination: str | None = None
        for ev in self._collisions:
            other = ev.other_actor.type_id if ev.other_actor is not None else "static"
            kind = (
                EventKind.COLLISION_VEHICLE
                if other.startswith("vehicle.")
                else EventKind.COLLISION_PEDESTRIAN
                if other.startswith("walker.")
                else EventKind.COLLISION_STATIC
            )
            events.append(Event(self.t, kind, other))
            termination = "collision"
        self._collisions.clear()

        for i, (ls, light) in enumerate(self._lights):
            if i in self._light_events_done:
                continue
            if prev_s < ls <= self._s and light.get_state() == carla.TrafficLightState.Red and self.v > RED_LIGHT_SPEED_MPS:
                events.append(Event(self.t, EventKind.RED_LIGHT, f"light {light.id}"))
                self._light_events_done.add(i)

        if self.junction_rules:
            events += self._serve_stop_signs()
        self._off_road_time = self._off_road_time + self.dt if abs(self._lat) > OFF_ROAD_LATERAL_M else 0.0
        if termination is None and self._off_road_time >= OFF_ROAD_SECONDS:
            events.append(Event(self.t, EventKind.OFF_ROAD, f"lateral {self._lat:.2f} m"))
            termination = "off_route"
        self._blocked_time = self._blocked_time + self.dt if self.v < BLOCKED_SPEED_MPS else 0.0
        if termination is None and self._blocked_time >= BLOCKED_SECONDS:
            events.append(Event(self.t, EventKind.BLOCKED, "stopped too long"))
            termination = "blocked"
        if termination is None and self._s >= self._route.length - ROUTE_END_MARGIN_M:
            termination = "route_complete"
        if termination is None and self.t >= self._scenario.max_time_s:
            termination = "timeout"

        self._done = termination is not None
        return StepResult(obs=self._observe(), truth=self.ground_truth(), done=self._done, events=tuple(events), termination=termination)

    def _cleanup(self) -> None:
        if self._npcs:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in self._npcs], True)
            self._npcs = []
        for a in reversed(self._actors):
            if a.type_id.startswith(("sensor.", "controller.")):
                a.stop()
            a.destroy()
        self._actors = []
        self._ego = None

    def close(self) -> None:
        """Destroys this episode's actors and restores asynchronous mode; the env can be reset again."""
        self._cleanup()
        self._scenario = None
        self.world.apply_settings(self._original_settings)
        self.tm.set_synchronous_mode(False)
