# 구조와 모듈 계약 (2026-10-21, 초안)

공통 규격과 각 모듈의 계약입니다. CARLA 어댑터(`carlautopilot/carla_env/`)는 2026-10-21에 데스크톱에서 CARLA 0.9.16으로 직접 확인하며 작성했습니다(아래 "CARLA 어댑터" 절).

## 원칙
- 환경(CARLA, 가짜 환경, 나중에 ETS2)은 `Env` 인터페이스 뒤에 숨기고, 정책·제어·평가는 환경을 몰라도 돌아가게 한다.
- **CARLA 전용 import는 `carlautopilot/carla_env/` 안에서만** 한다.
- 모든 모듈은 단위 테스트(pytest)를 갖고, 테스트가 통과한 것만 "동작한다"고 말한다.
- Python 3.10 이상에서 동작해야 한다. CARLA 0.9.16 클라이언트는 Python 3.10~3.12만 지원되므로 3.13 이상 전용 문법과 기능을 쓰지 않는다.
- 코드 주석은 최소로, 타입 힌트는 붙인다.

## 규약
- 단위: 미터, 초, m/s, 라디안. 좌표는 x가 동쪽, y가 북쪽. `yaw_rad`는 +x축에서 반시계 방향이 양수.
- `Action.steer`는 [-1, 1], **양수가 우회전**(차량이 시계 방향으로 돈다). 환경 어댑터가 각 환경의 규약으로 변환한다(CARLA 규약은 설치 후 확인).
- `Action.throttle`, `Action.brake`는 [0, 1].
- 경로 명령(`Command`): FOLLOW(0), LEFT(1), RIGHT(2), STRAIGHT(3). 시나리오의 `commands`는 경로를 따라간 거리 `s_m` 순으로 정렬한다. 현재 명령은 `s_m`이 현재 진행 거리 이하인 마지막 항목이고, 없으면 FOLLOW.
- `StepResult.termination`: `route_complete`, `timeout`, `collision`, `off_route`, `blocked`, 또는 `None`.
- 사건(`Event.kind`)은 `core.EventKind`의 문자열을 쓴다.
- 정책은 `Observation`만 본다. `GroundTruth`는 평가 전용이다(정책에 주지 않는다). 예외: 가짜 환경의 기준 정책 시험용으로 `Observation.extras`에 특권 정보를 넣는 옵션이 있다.

## 에피소드 로그 형식 (JSON Lines)
첫 줄은 `EpisodeHeader`(`scenario_name`, `seed`, `dt`, `policy_name`, `schema_version`), 이후 한 줄에 `StepRecord` 하나(`step`, `t`, `speed_mps`, `command`, `action`, `truth`, `events`, `lead_distance_m`, 모델 정책일 때 `percept`). `truth`에는 CARLA에서 다음 신호의 정지선까지 경로 거리와 빨간불(노란불 포함) 여부(`next_light_m`, `next_light_red`)가 들어간다. 두 필드는 2026-10-28에 더했고, 없는 옛 기록도 읽힌다. 쓰기와 읽기는 `carlautopilot/eval/episode_log.py`가 맡는다.

## 평가 지표 (프로젝트 정의, CARLA 리더보드 값이 아님)
- `route_completion` = clamp(최대 `route_progress_m` / `route_length_m`, 0, 1). 환경이 `route_complete`로 종료한 에피소드는 정확히 1.0으로 본다(환경이 경로 끝 근처의 허용 오차 안에서 완주로 판정하기 때문).
- 사건별 감점 계수(설정 가능, 기본값): collision_vehicle 0.6, collision_pedestrian 0.5, collision_static 0.65, red_light 0.7, stop_sign 0.8, off_road 0.8, lane_departure 0.95, route_deviation 0.8, blocked 0.9.
- `driving_score` = `route_completion` × (사건 하나당 계수의 곱).
- `success` = `termination == "route_complete"`이고 충돌 사건이 없다.
- 보조: 평균 속도, 최대 감속, 평균 |jerk|, 최소 앞차 간격, 소요 시간.
- 묶음 집계: 평균과 최소 driving_score, 성공률, 사건 종류별 횟수.

## 패키지 구조와 담당
| 경로 | 내용 | 비고 |
|---|---|---|
| `carlautopilot/core.py`, `scenario.py` | 공통 규격(관측, 행동, 이벤트, 환경·정책 인터페이스, 시나리오) | 완료, 변경은 사용자 승인 후 |
| `carlautopilot/sim/` | 가짜 환경 `FakeEnv` | 작업 A |
| `carlautopilot/control/` | ACC(IDM), PID, 순수 추종(pure pursuit), 안전 계층 | 작업 B |
| `carlautopilot/eval/` | 에피소드 로그, 지표, 집계, 보고서 | 작업 C |
| `carlautopilot/data/`, `models/`, `train/` | 데이터 형식, 데이터셋, 분할, 경로 조건부 조향 모델, 학습 루프 | 작업 D |
| `carlautopilot/policies/`, `carlautopilot/runner.py`, `carlautopilot/scenarios_lib.py`, `carlautopilot/cli/` | 기준 정책, 에피소드·묶음 실행기, 표준 시나리오, 명령줄 | 작업 E |

작업 A~E는 모두 구현되어 있다(2026-10-21, 직접 작성: Codex가 429로 거부해서). 현황은 `docs/plan.md`의 "구현 현황" 참고.
| `carlautopilot/carla_env/` | CARLA 어댑터(`env.py`), CARLA 없이 테스트하는 경로 보조 함수(`route.py`) | 2026-10-21 작성 |
| `tests/<영역>/` | 각 작업의 테스트 | 각 작업 |

## 모듈 계약

### A. `sim/fake_env.py` — FakeEnv
- `FakeEnv(dt=0.05, wheelbase_m=3.0, max_steer_rad=0.5, max_accel=3.0, max_brake_decel=8.0, drag=0.02, privileged_obs=False)`가 `Env`를 만족한다.
- 자전거 모델로 움직이고, 경로는 시나리오 `route` 폴리라인이다. 진행 거리 `s`와 횡오차를 폴리라인에 투영해 계산한다.
- 앞차(`LeadVehicleSpec`)는 경로를 따라 속도 프로파일대로 움직이고, 앞차와의 간격과 상대 속도를 `Observation.lead`로 준다(전방 100 m 이내일 때만).
- 신호등(`TrafficLightSpec`)은 시간에 따라 red/green이 바뀐다. 빨간불에서 정지선(`s_m`)을 속도 0.5 m/s 초과로 통과하면 `red_light` 사건을 신호등마다 한 번 낸다.
- 사건과 종료: 앞차 간격이 0 이하이면 `collision_vehicle` 후 `collision` 종료, 횡오차 2 m 초과가 2초 이상 이어지면 `off_road` 사건 후 `off_route` 종료, 속도 0.1 m/s 미만이 30초 이상이면 `blocked`, 시간 초과는 `timeout`, 진행 거리가 경로 길이 − 1 m 이상이면 `route_complete`.
- `reset(scenario, seed)`이 같은 입력이면 항상 같은 결과를 내야 한다(결정론적). `privileged_obs=True`이면 `extras`에 `heading_error_rad`, `lateral_error_m`, `lookahead_points`(앞쪽 경로점 목록)를 넣는다.
- `ground_truth()`와 `close()`를 제공한다.
- 구현에서 추가한 것: `privileged_obs`일 때 `extras`에 `next_light_distance_m`, `next_light_red`(앞쪽 100 m 이내에 신호등이 있을 때)도 넣는다. 앞차 속도 프로파일은 구간별 상수(시간 `t_k` 이후의 값)로 해석한다. `Scenario.meta["initial_speed_mps"]`로 시작 속도를 줄 수 있다.

### B. `control/`
- `idm.py`: `IDM`(a_max, b, v0, T, s0, delta)의 `accel(v, gap, dv)`(표준 IDM 식), 그리고 `ACCController`가 `Observation`과 속도 제한으로 목표 가속도를 구해 `(throttle, brake)`로 바꾼다. 앞차가 없으면 자유 주행 항만 쓴다.
- `pid.py`: 출력 제한과 적분 포화 방지가 있는 `PID(kp, ki, kd, out_min, out_max)`의 `step(error, dt)`와 `reset()`.
- `pure_pursuit.py`: `PurePursuit(wheelbase_m, lookahead_m, max_steer_rad)`가 특권 정보(`lookahead_points`, 현재 자세)로 조향 [-1, 1]을 낸다.
- `safety.py`: `SafetyConfig`와 `apply_safety(action, obs, cfg)`. 충돌까지 시간(TTC)이 임계값보다 작으면 브레이크를 건다. 앞차가 없으면 행동을 바꾸지 않는다.

### C. `eval/`
- `episode_log.py`: `EpisodeLogWriter`(컨텍스트 매니저, `write_header`, `write(record)`)와 `read_episode(path) -> (header, records)`.
- `metrics.py`: `compute_metrics(header, records, termination, penalties=None) -> dict`. 위 지표 정의를 그대로 따른다. 종료 사유는 로그에 따로 없으므로 마지막 레코드의 사건과 `route_progress_m`으로 추정하지 말고, 인자로 받는다.
- `aggregate.py`: 여러 에피소드의 지표 묶음을 집계한다.
- `report.py`: 집계 결과를 마크다운 보고서 문자열로 만든다.

### D. `data/`, `models/`, `train/`
- 에피소드 저장 형식: `<root>/<episode_id>/meta.json`(시나리오 이름, 시드, dt, 이미지 크기), `steps.npz`(`t`, `speed`, `command`, `steer`, `throttle`, `brake`), `images/000000.png` …(RGB). 쓰기·읽기 함수와 검증 함수를 제공한다.
- `episode_dataset.py`: PyTorch `Dataset`. 항목은 이미지(CHW float 0~1), 속도, 명령, 타깃 `[steer, throttle, brake]`.
- `splits.py`: **에피소드 단위** k-겹 분할(`make_folds(ids, k, seed)`). 같은 에피소드가 학습과 검증에 함께 들어가지 않는다.
- `models/route_steering.py`: 작은 CNN으로 이미지 특징을 뽑고, 속도 MLP와 합쳐, **명령별 4개의 출력 머리**(CIL 방식) 중 현재 명령의 것만 쓰는 `RouteSteeringNet`.
- `train/train_steering.py`: `train(cfg) -> dict`와 명령줄. 에폭, 배치, 학습률, 장치(cpu/cuda), 폴드를 받는다. 체크포인트와 지표 JSON을 쓴다. 이미 있는 출력 폴더는 덮어쓰지 않는다.

### E. 실행기와 기준 정책 (A~D 완료 후)
- `policies/`: `ConstantPolicy`, `PathFollowPolicy`(특권 정보 + IDM + 안전 계층).
- `runner.py`: `run_episode(env, policy, scenario, ...)`와 `run_suite(...)`가 로그와 지표, 보고서를 만든다.
- `scenarios_lib.py`: 직진, 곡선, 앞차 따라가기(앞차 감속), 빨간불 정지 시나리오 생성기.
- `cli/`: 묶음 평가 명령줄.
- 통합 시험: 기준 정책이 직진과 곡선을 완주하고, 앞차 감속에서 충돌하지 않고, 빨간불에서 서야 한다.

## CARLA 어댑터 (`carla_env/`, 2026-10-21 직접 확인한 사실)
- **조향 부호:** CARLA `VehicleControl.steer` 양수가 우회전이라 프로젝트 규약과 같다(steer=+0.5에서 yaw 0.2°→16.3°). 좌표는 CARLA가 왼손 좌표계(y가 오른쪽)라 `y`와 `yaw` 부호를 뒤집어 프로젝트 좌표로 바꾼다(`route.carla_to_xy`, `carla_yaw_to_rad`).
- **차량:** `vehicle.lincoln.mkz_2020`, 축거 2.86 m, 앞바퀴 최대 조향각 70°(조향 1.0 = 70°). 전문가의 순수 추종에 이 값을 쓴다.
- **동기 모드:** `fixed_delta_seconds = 0.05`, `world.tick()`마다 카메라(256×128, FOV 90, 차 앞 1.5 m·높이 1.6 m) 프레임 번호를 맞춰 받는다. Traffic Manager도 동기 모드로 둔다.
- **경로:** 차로 변경(`CHANGELANELEFT/RIGHT`)이 들어간 경로는 버린다(모델에 차로 변경 명령이 없어서). CARLA 패키지의 `agents.navigation.GlobalRoutePlanner`(2 m 간격, `networkx` 필요, `--carla-root`로 경로 지정)로 출발점→도착점 경로를 만든다. 교차로 구간의 `RoadOption`(LEFT/RIGHT/STRAIGHT)을 교차로 10 m 앞부터 명령으로 주고, 교차로를 지나면 FOLLOW. 자기 경로 근처를 다시 지나는 경로 때문에 진행 거리 투영은 현재 위치 주변 구간으로 제한한다(`WindowedRoute`).
- **앞 장애물:** 경로 복도(중심선 좌우 1.8 m) 안의 차량·보행자 중 가장 가까운 것의 차체 간 거리와 경로 방향 상대 속도. 시뮬레이터의 정답 위치로 계산하므로 레이더 같은 센서를 흉내 낸 **특권 정보**다.
- **신호등:** 각 신호등의 정지 웨이포인트가 경로 위(2 m 이내, 방향 30° 이내)에 있으면 그 진행 거리를 정지선으로 쓴다. 빨간불에 0.5 m/s 초과로 정지선을 넘으면 `red_light`. 전문가는 빨간불과 노란불에서 선다.
- **종료:** 충돌 센서의 모든 충돌(차량·보행자·고정물), 횡오차 2 m 초과 2초(`off_route`), 0.1 m/s 미만 **90초**(`blocked`, CARLA 신호 대기가 30초를 넘는 일이 있어서 FakeEnv의 30초보다 길게), 경로 끝 2 m 전 도달(`route_complete`), 시간 초과(경로 길이/속도×3+120초).
- **보조 카메라(`extra_views`):** (왼쪽 거리 m, 왼쪽 회전 °) 목록. 수집 때 각 카메라 영상에 "차가 그 위치·방향에 있었다면 전문가가 냈을 조향"을 라벨로 붙인다(`route.points_seen_from`로 앞쪽 경로점을 옮겨 다시 계산). 페달 라벨은 중앙 카메라와 같다.
- **관전 시점(`follow_spectator`, 기본 켜짐):** CARLA 창의 관전자 카메라를 매 틱 우리 차 뒤 8 m·위 4 m에서 15° 내려다보게 옮긴다. 학습용 카메라와 별개라 데이터에 영향이 없다.
- **차종(`vehicle`):** `VEHICLES` 표(청사진, 축거, 최대 조향각, 카메라 위치). sedan = 링컨 MKZ(2.86 m, 카메라 1.5 m·1.6 m), truck = `european_hgv`(4.42 m, 카메라 4.1 m·2.4 m). CARLA 0.9.16에는 트레일러가 없다.
- **카메라 묶음(`rigs`):** (시야각, 회전 카메라 각도) 목록. 묶음 0이 `obs.image`, 나머지는 `extras["rig_images"]`. 회전 카메라 각도가 0이 아니면 그 묶음 전체가 경로 명령 방향(LEFT +, RIGHT -)으로 `TURN_CAMERA_RATE_DEG_S`(25°/s)씩 돈다. 차의 자세는 그대로라 라벨은 묶음과 상관없이 같다. 수집 때 `--rigs`를 주면 묶음마다 `<data-root>_fov<F>_turn<T>`에 같은 라벨로 저장한다. 평가는 `--fov`, `--turn-camera`로 같은 카메라를 쓴다.
- **속도 옵션(데이터 무손실):** 수집 `--png-compress`(기본 6, 1이 빠름·파일 1.5배), CARLA `-RenderOffScreen`. 학습 `--cache-dir`(PNG를 한 번 풀어 둔 .npy, 픽셀 동일) + `--num-workers 0`. 수집 기록에 `sim_s`/`write_s`가 남는다.
- **맵 전환:** 같은 프로세스에서 `load_world`를 하면 클라이언트가 간헐적으로 강제 종료됐다(`_episode != nullptr` assert, Simulator.h). 그래서 `switch_map()`이 별도 프로세스에서 맵을 바꾸고, 수집·평가 명령은 한 번에 도시 하나만 다룬다.
- **명령줄:** `python -m carlautopilot.cli.carla_run collect|eval --carla-root <CARLA 폴더> --town <도시> ...`. 수집은 전문가(`PathFollowPolicy`, 회전 감속 `max_lat_accel=2.0`, ACC 최소 간격 4 m)에 조향 잡음(기본 0.05)을 넣어 달리고 깨끗한 전문가 행동을 라벨로 저장한다. 평가는 `--checkpoint`가 없으면 전문가, 있으면 모델(`--model-pedals`가 없으면 페달은 ACC, 기본 속도 6 m/s, 교차로 명령 중 4.5 m/s 규칙)을 `run_suite`로 돌린다.
