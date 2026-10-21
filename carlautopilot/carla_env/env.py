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

from carlautopilot.carla_env.route import LANE_CHANGE_OPTIONS, WindowedRoute, carla_to_xy, carla_yaw_to_rad, commands_from_options, dedupe_points
from carlautopilot.core import Action, Command, Event, EventKind, GroundTruth, LeadInfo, Observation, StepResult
from carlautopilot.scenario import RouteCommand, ScenarioSpec, Waypoint
from carlautopilot.sim.fake_env import (
    BLOCKED_SPEED_MPS,
    LEAD_VISIBLE_M,
    LOOKAHEAD_OFFSETS_M,
    OFF_ROAD_LATERAL_M,
    OFF_ROAD_SECONDS,
    RED_LIGHT_SPEED_MPS,
    _wrap,
)

VEHICLE = "vehicle.lincoln.mkz_2020"
WHEELBASE_M = 2.86
MAX_STEER_RAD = math.radians(70.0)
ROUTE_END_MARGIN_M = 2.0
BLOCKED_SECONDS = 90.0  # CARLA reds at 4-way junctions can exceed 30 s
CORRIDOR_HALF_M = 1.8
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
        extra_views: tuple[tuple[float, float], ...] = (),
        follow_spectator: bool = True,
    ):
        """extra_views: additional cameras as (offset left m, yaw left deg); their images go to extras["view_images"]."""
        self.dt = dt
        self.extra_views = tuple(extra_views)
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
        self._queues: list[queue.Queue] = []
        self._last_image: np.ndarray | None = None
        self._view_images: list[np.ndarray] = []

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
            opts = [int(trace[i][1]) for i in keep]
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

        self._queues: list[queue.Queue] = []
        if self.render_images:
            cam_bp = bpl.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.image_size[1]))
            cam_bp.set_attribute("image_size_y", str(self.image_size[0]))
            cam_bp.set_attribute("fov", "90")
            for dy, dyaw in ((0.0, 0.0),) + self.extra_views:
                mount = carla.Transform(carla.Location(x=1.5, y=-dy, z=1.6), carla.Rotation(yaw=-dyaw))
                cam = self.world.spawn_actor(cam_bp, mount, attach_to=self._ego)
                q: queue.Queue = queue.Queue()
                cam.listen(q.put)
                self._queues.append(q)
                self._actors.append(cam)
        self._collisions: list = []
        col = self.world.spawn_actor(bpl.find("sensor.other.collision"), carla.Transform(), attach_to=self._ego)
        col.listen(self._collisions.append)
        self._actors.append(col)

        self._spawn_npcs(rng, int(meta.get("npc_vehicles", 0)), start.location)

        self._scenario = scenario
        self._route = WindowedRoute([(w.x, w.y) for w in scenario.route])
        self._lights = self._lights_on_route()
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

    def _tick(self) -> None:
        frame = self.world.tick()
        images = []
        for q in self._queues:
            while True:
                img = q.get(timeout=10.0)
                if img.frame >= frame:
                    break
            a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            images.append(np.ascontiguousarray(a[:, :, 2::-1]))
        if images:
            self._last_image, self._view_images = images[0], images[1:]

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
            s, lat, _ = self._route.project(ox, oy, self._s, self._s + LEAD_VISIBLE_M)
            if abs(lat) > CORRIDOR_HALF_M or s <= self._s:
                continue
            gap = s - self._s - ego_half - a.bounding_box.extent.x
            if gap > LEAD_VISIBLE_M:
                continue
            h = self._route.heading_at(s)
            vel = a.get_velocity()
            vx, vy = carla_to_xy(vel.x, vel.y)
            along = vx * math.cos(h) + vy * math.sin(h)
            if best is None or gap < best.distance_m:
                best = LeadInfo(distance_m=max(gap, 0.0), rel_speed_mps=along - self.v)
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
            "view_images": self._view_images,
            "heading_error_rad": _wrap(self.yaw - self._route.heading_at(self._s)),
            "lateral_error_m": self._lat,
            "lookahead_points": pts,
        }
        for ls, light in self._lights:
            dist = ls - self._s
            if 0.0 < dist <= LEAD_VISIBLE_M:
                extras["next_light_distance_m"] = dist
                extras["next_light_red"] = light.get_state() in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)
                break
        return extras

    def _observe(self) -> Observation:
        image = self._last_image if self.render_images else None
        return Observation(t=self.t, speed_mps=self.v, command=self._command(), image=image, lead=self._lead_info(), extras=self._extras())

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(self.x, self.y, self.yaw, self.v, self._s, self._route.length, self._lat)

    def step(self, action: Action) -> StepResult:
        if self._scenario is None:
            raise RuntimeError("call reset() first")
        if self._done:
            raise RuntimeError("episode finished; call reset()")
        a = action.clipped()
        self._ego.apply_control(carla.VehicleControl(throttle=a.throttle, steer=a.steer, brake=a.brake))
        prev_s = self._s
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
            if a.type_id.startswith("sensor."):
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
