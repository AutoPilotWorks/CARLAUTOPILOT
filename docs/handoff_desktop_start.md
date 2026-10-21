# 집 데스크톱 시작 프롬프트 (2026-10-21)

아래 블록을 집 데스크톱(RTX 4070 SUPER)의 Claude Code 첫 메시지에 그대로 붙여 넣습니다. 이 저장소를 새 폴더에 받은 뒤 그 폴더에서 새 세션을 여세요.

````text
CARLAUTOPILOT 프로젝트를 이 집 데스크톱(RTX 4070 SUPER 12GB)에서 시작합니다. 한국어 존댓말로, 작은 단계로, 실행할 명령어와 확인할 결과를 구체적으로 알려주세요. 코드 주석은 최소화해주세요.

1. 저장소 AutoPilotWorks/CARLAUTOPILOT(비공개, main)이 이 컴퓨터에 없다면 사용자가 정하는 폴더에 받아주세요: `git clone https://github.com/AutoPilotWorks/CARLAUTOPILOT.git`. 받은 뒤 `git log --oneline -3`을 보여주세요.
2. `CLAUDE.md`, `docs/working_rules.md`, `docs/plan.md`를 순서대로 끝까지 읽어주세요. 특히 plan.md의 "확정된 결정"과 "아직 확정되지 않은 것"을 구분해서 이해해주세요.
3. 읽은 내용을 5줄 이내로 요약해 제가 맞는지 확인할 수 있게 해주세요.
4. 환경을 확인만 해주세요(설치는 하지 마세요). 저장소를 받은 뒤에는 `python -m carlautopilot.cli.check_env`로 한 번에 점검할 수 있습니다(보고만 하고 설치하지 않음; 개별 항목은 아래):
   - `git --version`, `python --version`, `nvidia-smi`(GPU 이름, VRAM, CUDA 버전).
   - 디스크 여유 공간(CARLA는 약 20GB가 필요합니다).
   - Python 3.12가 설치되어 있는지(CARLA Python API는 3.12까지 지원합니다. 없으면 기존 Python과 나란히 설치하는 방법만 알려주세요).
5. (CARLA와 무관한 확인) 저장소에 이미 구현된 코드가 있습니다. Python 3.12 가상환경을 만들 수 있다면 `pip install -r requirements.txt`와 CPU 또는 CUDA용 `torch`, `Pillow`를 설치하고 `python -m pytest -q`를 돌려 결과를 알려주세요. 이전 컴퓨터(Python 3.14)에서는 121개가 통과했지만 3.12에서는 아직 확인하지 못했습니다. 설치가 필요하면 먼저 승인을 받으세요.
6. 환경 확인 결과를 알려주고 멈춰주세요. CARLA 설치, 가상환경 생성, 코드 작성, 학습, push는 제가 승인하기 전에는 시작하지 마세요. 설치 단계에서는 공식 문서(https://carla.readthedocs.io)로 요구 사양과 절차를 직접 확인하면서 진행하고, 확인하지 못한 내용을 확인했다고 말하지 마세요.

참고: 이전 프로젝트 ets2-autopilot-learning(비공개)은 참고용입니다. 그 코드는 쓸 만하다고 판단되면 가져와도 되지만, ETS2 데이터셋은 비상업적 연구용이라 이 프로젝트로 가져오지 않습니다. 이 저장소에서는 Claude 세션을 한 번에 하나만 사용해주세요.
````

## 이어서 할 첫 작업 (사용자 승인 후)
1. CARLA 0.9.15 설치와 Python 3.12 가상환경 준비(공식 문서 기준).
2. CARLA 서버 실행, 카메라 영상 저장 확인.
3. 기본 자동 주행으로 데이터(영상, 조향, 경로 명령) 자동 수집이 되는지 확인.
4. 첫 2주 안에 위 세 가지가 되는지 점검하고, 안 되면 방향을 다시 판단한다([계획](plan.md)).
