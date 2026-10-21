import pytest

from carlautopilot.control import IDM, PID, ACCController, PurePursuit, SafetyConfig, apply_safety, time_to_collision
from carlautopilot.core import Action, Command, LeadInfo, Observation


def obs(speed=10.0, lead=None):
    return Observation(t=0.0, speed_mps=speed, command=Command.FOLLOW, lead=lead)


def test_idm_free_road():
    m = IDM(a_max=1.5, v0=14.0)
    assert m.accel(0.0) == pytest.approx(1.5)
    assert m.accel(7.0) > 0
    assert m.accel(14.0) == pytest.approx(0.0, abs=1e-9)
    assert m.accel(10.0) <= 1.5


def test_idm_equilibrium_gap():
    m = IDM()
    v = 8.0
    gap = m.equilibrium_gap(v)
    assert m.accel(v, gap, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_idm_brakes_when_closing_on_close_lead():
    m = IDM()
    assert m.accel(14.0, 10.0, 8.0) < -4.0
    assert m.accel(5.0, 100.0, 0.0) > 0


def test_idm_closing_speed_sign():
    m = IDM()
    slower_lead = m.accel(10.0, 30.0, 5.0)
    faster_lead = m.accel(10.0, 30.0, -5.0)
    assert slower_lead < faster_lead


def test_acc_free_and_follow():
    acc = ACCController()
    th, br = acc.act(obs(speed=0.0), speed_limit=13.9)
    assert th == pytest.approx(1.0) and br == 0.0
    th, br = acc.act(obs(speed=13.9), speed_limit=13.9)
    assert th == pytest.approx(0.0, abs=1e-9) and br == 0.0
    th, br = acc.act(obs(speed=12.0, lead=LeadInfo(8.0, -6.0)), speed_limit=13.9)
    assert th == 0.0 and 0.0 < br <= 1.0


def test_acc_respects_speed_limit():
    acc = ACCController()
    th, br = acc.act(obs(speed=10.0), speed_limit=5.0)
    assert th == 0.0 and br > 0.0


def test_pid_step_clamp_reset():
    pid = PID(kp=2.0, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0)
    assert pid.step(0.25, 0.1) == pytest.approx(0.5)
    assert pid.step(5.0, 0.1) == 1.0
    assert pid.step(-5.0, 0.1) == -1.0
    with pytest.raises(ValueError):
        pid.step(0.0, 0.0)
    with pytest.raises(ValueError):
        PID(1, 0, 0, 1.0, 1.0)


def test_pid_no_derivative_kick_and_derivative():
    pid = PID(kp=0.0, ki=0.0, kd=1.0, out_min=-100, out_max=100)
    assert pid.step(1.0, 0.1) == 0.0
    assert pid.step(2.0, 0.1) == pytest.approx(10.0)
    pid.reset()
    assert pid.step(5.0, 0.1) == 0.0


def test_pid_antiwindup():
    pid = PID(kp=1.0, ki=1.0, kd=0.0, out_min=-1.0, out_max=1.0)
    for _ in range(100):
        pid.step(10.0, 0.1)
    assert pid._integral < 5.0
    assert pid.step(-10.0, 0.1) < 1.0


def test_pid_integrates_when_unsaturated():
    pid = PID(kp=0.0, ki=1.0, kd=0.0, out_min=-10, out_max=10)
    pid.step(1.0, 0.5)
    assert pid.step(1.0, 0.5) == pytest.approx(1.0)


def test_pure_pursuit_signs():
    pp = PurePursuit(wheelbase_m=3.0, lookahead_m=8.0, max_steer_rad=0.5)
    assert pp.steer([(8.0, 0.0)]) == pytest.approx(0.0)
    assert pp.steer([(8.0, 2.0)]) < 0
    assert pp.steer([(8.0, -2.0)]) > 0
    assert pp.steer([(8.0, 2.0)]) == pytest.approx(-pp.steer([(8.0, -2.0)]))
    assert -1.0 <= pp.steer([(1.0, 50.0)]) <= 1.0


def test_pure_pursuit_picks_lookahead_point():
    pp = PurePursuit(lookahead_m=8.0)
    assert pp.target([(2.0, 0.0), (5.0, 1.0), (9.0, 0.0), (20.0, 5.0)]) == (9.0, 0.0)
    assert pp.target([(2.0, 0.0), (3.0, 0.0)]) == (3.0, 0.0)
    with pytest.raises(ValueError):
        pp.target([])


def test_time_to_collision():
    assert time_to_collision(obs()) is None
    assert time_to_collision(obs(lead=LeadInfo(20.0, 1.0))) is None
    assert time_to_collision(obs(lead=LeadInfo(20.0, -4.0))) == pytest.approx(5.0)


def test_safety_cases():
    cfg = SafetyConfig(ttc_brake_s=2.5, ttc_full_brake_s=1.0)
    a = Action(0.1, 0.8, 0.0)
    assert apply_safety(a, obs(), cfg) == a
    assert apply_safety(a, obs(lead=LeadInfo(100.0, -5.0)), cfg) == a
    braked = apply_safety(a, obs(lead=LeadInfo(10.0, -5.0)), cfg)
    assert braked.throttle == 0.0 and 0.0 < braked.brake < 1.0 and braked.steer == 0.1
    full = apply_safety(a, obs(lead=LeadInfo(2.0, -5.0)), cfg)
    assert full.brake == 1.0 and full.throttle == 0.0
    keep = apply_safety(Action(0.0, 0.0, 0.9), obs(lead=LeadInfo(10.0, -5.0)), cfg)
    assert keep.brake >= 0.9
    wild = apply_safety(Action(5.0, 3.0, -1.0), obs(), cfg)
    assert (wild.steer, wild.throttle, wild.brake) == (1.0, 1.0, 0.0)
