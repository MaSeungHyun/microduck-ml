# Microduck 강화학습 훈련 프로젝트

Pollen Robotics의 Microduck 이족보행 로봇을 위한 강화학습 훈련 프로젝트입니다.

## 로봇 정보

**Microduck**은 Pollen Robotics에서 개발한 소형 이족보행 로봇입니다.

- **높이**: ~25cm
- **무게**: ~800g
- **자유도**: 14 DOF (액추에이터)
  - 왼쪽 다리: hip_yaw, hip_roll, hip_pitch, knee, ankle (5개)
  - 머리: neck_pitch, head_pitch, head_yaw, head_roll (4개)
  - 오른쪽 다리: hip_yaw, hip_roll, hip_pitch, knee, ankle (5개)
- **모터**: Dynamixel XL330 (15개)

### 로봇 모델 출처

- **저장소**: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
- **브랜치**: develop
- **라이선스**:
  - STL 메시 파일: CC BY-NC-SA 4.0 (Pollen Robotics)
  - MJCF XML 파일: Apache 2.0 (Pollen Robotics)

## 시스템 요구사항

- **OS**: Windows 11 / Linux
- **GPU**: NVIDIA RTX 4060 (8GB VRAM) 이상 권장
- **Python**: 3.11
- **CUDA**: 12.1 이상

## 설치

```bash
# 저장소 클론
git clone <repository-url>
cd microduck-rl

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux
# 또는
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### Genesis 설치 (GPU 가속)

GPU 가속 시뮬레이션을 위해 Genesis를 설치합니다:

```bash
pip install genesis-world==1.4.5
```

> **참고**: Genesis는 NVIDIA GPU와 CUDA가 필요합니다.

## 커리큘럼 기반 훈련

훈련은 3단계 커리큘럼으로 진행됩니다:

### 1단계: STAND (서기)

로봇이 균형을 잡고 서있는 것을 학습합니다.

```bash
python train.py --stage stand --num-envs 2048 --iterations 1000
```

### 2단계: WALK (걷기)

전방으로 걷는 것을 학습합니다.

```bash
python train.py --stage walk --num-envs 2048 --iterations 2000 --resume checkpoints/stand_best.pt
```

### 3단계: COMMAND (속도 명령)

임의의 속도 명령을 추종하면서 도메인 랜덤화를 적용합니다.

```bash
python train.py --stage command --num-envs 2048 --iterations 3000 --resume checkpoints/walk_best.pt
```

## 훈련 옵션

```bash
python train.py --help
```

주요 옵션:
- `--stage`: 커리큘럼 단계 (stand/walk/command)
- `--num-envs`: 병렬 환경 수 (기본: 2048)
- `--iterations`: 훈련 반복 횟수
- `--resume`: 체크포인트에서 이어서 훈련
- `--backend`: 시뮬레이션 백엔드 (genesis/mujoco)
- `--device`: 디바이스 (cuda/cpu)
- `--render`: 실시간 렌더링 활성화
- `--seed`: 랜덤 시드

## 정책 평가

훈련된 정책을 평가하고 시각화합니다:

```bash
# Genesis 백엔드 (GPU)
python play.py --checkpoint checkpoints/stand_best.pt --render

# MuJoCo 백엔드 (CPU, 디버깅용)
python play.py --checkpoint checkpoints/stand_best.pt --backend mujoco --render

# 헤드리스 평가
python play.py --checkpoint checkpoints/stand_best.pt --episodes 100
```

## 프로젝트 구조

```
microduck-rl/
├── assets/
│   ├── microduck.xml      # Microduck MJCF 로봇 모델
│   ├── scene.xml          # 바닥이 포함된 시뮬레이션 장면
│   └── meshes/            # STL 메시 파일 (43개)
│       ├── body.stl
│       ├── left_hip_yaw_horn.stl
│       └── ...
├── envs/
│   └── microduck_env.py   # Gymnasium 스타일 환경
├── train.py               # PPO 훈련 스크립트
├── play.py                # 평가/시각화 스크립트
├── test_env.py            # 환경 테스트 스크립트
├── requirements.txt       # 파이썬 패키지 의존성
├── LICENSE                # MIT + 서드파티 라이선스
└── README.md              # 이 파일
```

## 관찰 공간

환경의 관찰 벡터 (53차원):
- 관절 위치 (14)
- 관절 속도 (14)
- 기저부 각속도 (3)
- 중력 벡터 (3)
- 속도 명령 (3)
- 이전 행동 (14)
- 보행 위상 (2)

## 행동 공간

14차원 연속 행동 공간 ([-1, 1] 범위):
- 각 관절의 목표 위치 오프셋
- 기본 자세에서의 스케일된 오프셋

## 보상 구조

- `alive`: 생존 보상 (에피소드 지속)
- `upright`: 직립 자세 유지
- `height`: 목표 높이 유지
- `velocity_tracking`: 속도 명령 추종
- `angular_velocity_tracking`: 회전 속도 명령 추종
- `action_rate`: 행동 변화율 페널티
- `joint_torque`: 관절 토크 페널티

## 종료 조건

- 기저부 높이가 5cm 미만
- 롤/피치 각도가 1.2 rad (~70°) 초과

## 문제 해결

### Genesis 설치 오류

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"
```

### MuJoCo 렌더링 오류

```bash
# 필요한 라이브러리 설치 (Linux)
sudo apt-get install libglfw3 libglew2.1
```

### 메모리 부족

병렬 환경 수를 줄이세요:
```bash
python train.py --num-envs 512
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

**중요**: Microduck 로봇 모델의 STL 메시 파일은 **CC BY-NC-SA 4.0** 라이선스입니다.
상업적 사용 시 Pollen Robotics에 문의하세요.

## 참고 자료

- [Pollen Robotics](https://www.pollen-robotics.com/)
- [Microduck RL Repository](https://github.com/pollen-robotics/microduck_rl)
- [Genesis Simulator](https://genesis-embodied-ai.github.io/)
- [MuJoCo](https://mujoco.org/)

## 기여

버그 리포트, 기능 요청, PR을 환영합니다!
