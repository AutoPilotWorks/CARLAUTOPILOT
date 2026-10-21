# CARLAUTOPILOT — CARLA 기반 영상 자율주행 연구

CARLA 시뮬레이터에서 **영상으로 판단하는 자율주행**을 만들고, 거의 모든 기능이 동작한다고 판단되면 Euro Truck Simulator 2(ETS2)로 옮기는 프로젝트입니다. 최종 목표는 화물을 배달하는 경로를 처음부터 끝까지 스스로 주행하는 것입니다.

## 현재 상태 (2026-10-21)
- **계획 단계입니다. 코드는 아직 없습니다.** 문서와 작업 규칙만 있습니다.
- 첫 작업은 집 데스크톱에서 CARLA 0.9.15와 Python 3.12 환경을 준비하는 것입니다([시작 프롬프트](docs/handoff_desktop_start.md)).

## 왜 별도 프로젝트인가
- 이전 프로젝트 [ets2-autopilot-learning](https://github.com/AutoPilotWorks/ets2-autopilot-learning)(비공개)는 ETS2 게임 화면으로 영상 기반 조향·감속 모델, 경로 조건부 조향, 안전 계층, 가상 컨트롤러 주행까지 진행했습니다.
- 게임은 같은 상황을 반복해서 시험할 수 없고 정답(주변 물체의 정확한 위치 등)이 없어서, 정답이 있고 재현되는 시뮬레이터에서 기능을 먼저 완성하기로 했습니다([결정 기록](docs/plan.md)).
- ETS2 프로젝트의 기록과 도구는 그대로 보존되며, 이 프로젝트에서 필요한 것만 골라 가져옵니다. ETS2 **데이터셋**은 비상업적 연구용 조건이라 가져오지 않습니다.

## 문서
- [계획과 확정된 결정](docs/plan.md)
- [작업 규칙](docs/working_rules.md) — 새 세션이나 새 컴퓨터에서 일하기 전에 먼저 읽어 주세요.
- [데스크톱 시작 프롬프트](docs/handoff_desktop_start.md)
- Claude Code 작업 규칙: [CLAUDE.md](CLAUDE.md)

## 환경 (계획)
- 실행: 집 데스크톱(RTX 4070 SUPER 12GB). 코드 작성과 GPU가 필요 없는 작업은 다른 컴퓨터에서도 합니다.
- CARLA 0.9.15, Python 3.12(CARLA Python API는 3.12까지 지원). 요구 사양은 공식 문서 기준 VRAM 8GB 이상 권장, 디스크 약 20GB.
