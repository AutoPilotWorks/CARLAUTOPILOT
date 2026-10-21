# 구조와 모듈 계약 (2026-10-21, 초안)

CARLA 없이 먼저 만들 수 있는 부분의 설계입니다. **CARLA 연결 코드는 아직 없습니다.** 그 부분은 CARLA를 설치하고 공식 문서로 확인한 뒤에 만듭니다(`carlautopilot/carla_env/`, 미작성).

## 원칙
- 환경(CARLA, 가짜 환경, 나중에 ETS2)은 `Env` 인터페이스 뒤에 숨기고, 정책·제어·평가는 환경을 몰라도 돌아가게 한다.
- **CARLA 전용 import는 `carlautopilot/carla_env/` 안에서만** 한다.
- 모든 모듈은 단위 테스트(pytest)를 갖고, 테스트가 통과한 것만 "동작한다"고 말한다.
- Python 3.10 이상에서 동작해야 한다. CARLA용 Python은 3.12까지만 지원되므로 3.13 이상 전용 문법과 기능을 쓰지 않는다.
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
첫 줄은 `EpisodeHeader`(`scenario_name`, `seed`, `dt`, `policy_name`, `schema_version`), 이후 한 줄에 `StepRecord` 하나(`step`, `t`, `speed_mps`, `command`, `action`, `truth`, `events`, `lead_distance_m`). 쓰기와 읽기는 `carlautopilot/eval/episode_log.py`가 맡는다.

## 평가 지표 (프로젝트 정의, CARLA 리더보드 값이 아님)
- `route_completion` = clamp(최대 `route_progress_m` / `route_length_m`, 0, 1).
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
| `carlautopilot/policies/`, `carlautopilot/runner.py`, `carlautopilot/scenarios_lib.py`, `carlautopilot/cli/` | 기준 정책, 에피소드·묶음 실행기, 표준 시나리오, 명령줄 | 작업 E(A~D 완료 후) |
| `carlautopilot/carla_env/` | CARLA 어댑터 | **미작성**(CARLA 설치 후) |
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

## CARLA 어댑터에서 확인할 것 (설치 후)
- 서버 연결, 동기 모드와 고정 시간 간격, 센서(카메라) 부착과 영상 수신, 차량 생성과 제어(조향 부호 규약 포함).
- 경로(웨이포인트) 생성과 명령 부여, 신호등·앞차 상태 조회, 충돌·차선 침범 센서.
- 기본 자동 주행(Traffic Manager)으로 데이터 수집.
