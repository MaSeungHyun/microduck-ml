# 🦆 Microduck - 이족보행 로봇 강화학습 프로젝트

작은 이족보행 로봇 "Microduck"을 강화학습(RL)으로 걷게 만드는 프로젝트입니다.  
MuJoCo와 Genesis 물리 시뮬레이터를 사용하여 GPU 기반 대규모 병렬 학습을 지원합니다.

> **크레딧**: 이 프로젝트는 [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) (Apache 2.0 라이선스)와 [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground)에서 영감을 받았습니다.

---

## 📋 목차

- [프로젝트 구조](#프로젝트-구조)
- [설치 방법 (Windows 11)](#설치-방법-windows-11)
- [사용 방법](#사용-방법)
- [커리큘럼 학습 단계](#커리큘럼-학습-단계)
- [보상 함수 설명](#보상-함수-설명)
- [하이퍼파라미터 조정](#하이퍼파라미터-조정)
- [문제 해결](#문제-해결)

---

## 📁 프로젝트 구조

```
microduck/
├── assets/
│   └── microduck.xml      # 로봇 MJCF 모델 파일
├── envs/
│   ├── __init__.py
│   └── microduck_env.py   # Gymnasium 환경 구현
├── scripts/
│   ├── train.py           # PPO 학습 스크립트
│   └── play.py            # 학습된 정책 평가/시각화
├── runs/                   # 학습 결과 저장 폴더
│   └── <실험명>/
│       ├── model_*.pt     # 체크포인트
│       └── progress.csv   # 학습 진행 로그
├── requirements.txt       # Python 의존성
└── README.md              # 이 문서
```

---

## 🛠 설치 방법 (Windows 11)

### 요구 사항

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM) 이상
- **Python**: 3.11.x
- **CUDA**: 12.x

### 1단계: Python 가상환경 생성

PowerShell을 **관리자 권한**으로 실행하고:

```powershell
# 프로젝트 폴더로 이동
cd C:\path\to\microduck

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
.\venv\Scripts\Activate

# pip 업그레이드
python -m pip install --upgrade pip
```

### 2단계: PyTorch 설치 (CUDA 12.8)

```powershell
# CUDA 12.8 버전 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **참고**: CUDA 12.1을 사용하는 경우 `cu128` 대신 `cu121`을 사용하세요.

### 3단계: 나머지 패키지 설치

```powershell
# Genesis 및 기타 의존성
pip install genesis-world numpy gymnasium tensorboard tqdm matplotlib

# MuJoCo (선택사항, CPU 디버깅용)
pip install mujoco
```

### 4단계: 설치 확인

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import genesis; print('Genesis OK')"
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
```

---

## 🎮 사용 방법

### 기본 학습 시작

```powershell
# 1단계: 균형 잡기 학습 (약 500 반복)
python scripts/train.py --stage stand --num-envs 1024 --iterations 500

# 2단계: 걷기 학습 (약 5000 반복, 1단계 체크포인트 사용)
python scripts/train.py --stage walk --num-envs 1024 --iterations 5000 --checkpoint runs/microduck_stand_*/model_final.pt

# 3단계: 속도 명령 추적 학습 (약 10000 반복)
python scripts/train.py --stage command --num-envs 1024 --iterations 10000 --checkpoint runs/microduck_walk_*/model_final.pt
```

### 라이브 뷰어로 학습 (느리지만 시각적)

```powershell
# 3x3 그리드로 9개 환경 시각화하며 학습
python scripts/train.py --stage stand --render --viewer-grid 3x3 --iterations 100
```

### 학습된 정책 평가

```powershell
# 최신 체크포인트로 평가
python scripts/play.py

# 특정 체크포인트 지정
python scripts/play.py --checkpoint runs/microduck_walk_*/model_final.pt --stage walk
```

### GPU 메모리 관리 (8GB VRAM)

RTX 4060 8GB의 경우 권장 설정:

| 단계 | 환경 수 | 예상 VRAM |
|------|---------|-----------|
| stand | 1024 | ~3-4 GB |
| walk | 1024 | ~4-5 GB |
| command | 512 | ~4-5 GB |

```powershell
# 메모리가 부족하면 환경 수 줄이기
python scripts/train.py --stage stand --num-envs 512 --iterations 500
```

---

## 📚 커리큘럼 학습 단계

이 프로젝트는 **커리큘럼 학습**을 사용합니다. 쉬운 것부터 어려운 것 순서로 단계별로 학습합니다.

### 1단계: Stand (균형 잡기) 🧍

**목표**: 쓰러지지 않고 제자리에 서있기

- 속도 명령: 없음 (정지 상태)
- 외부 힘: 없음
- 도메인 랜덤화: 비활성화

**성공 기준**:
- 500번 반복 후 평균 보상 ~25 이상
- 에피소드 길이가 최대치(1000 스텝)에 도달

### 2단계: Walk (걷기) 🚶

**목표**: 앞으로 일정한 속도로 걷기

- 속도 명령: 전방 0.2~0.4 m/s
- 외부 힘: 없음
- 도메인 랜덤화: 비활성화

**성공 기준**:
- 5000번 반복 후 평균 보상 ~50 이상
- 로봇이 안정적으로 앞으로 이동

### 3단계: Command (명령 추적) 🎯

**목표**: 다양한 속도 명령을 추적하며 걷기

- 속도 명령: 
  - X (전/후): -0.2 ~ 0.5 m/s
  - Y (좌/우): -0.2 ~ 0.2 m/s
  - Yaw (회전): -0.5 ~ 0.5 rad/s
- 외부 힘: 5~20N 랜덤 밀기
- 도메인 랜덤화: 활성화 (질량, 마찰 변화)

**성공 기준**:
- 10000번 반복 후 평균 보상 ~70 이상
- 다양한 명령에 부드럽게 반응

---

## 🏆 보상 함수 설명

강화학습에서 **보상(Reward)**은 "로봇이 얼마나 잘하고 있는지"를 숫자로 알려주는 점수입니다.

### 보상 항목 상세

| 보상 이름 | 가중치 | 설명 | 초보자 설명 |
|-----------|--------|------|-------------|
| **alive** | +2.0 | 쓰러지지 않으면 받는 보상 | "살아있으면 점수!" |
| **upright** | +1.0 | 몸이 똑바로 서있을수록 높음 | "비뚤어지면 감점" |
| **height** | +1.0 | 목표 높이(22cm)에 가까울수록 높음 | "너무 웅크리면 감점" |
| **velocity_tracking** | +2.0 | 명령된 속도와 실제 속도가 비슷할수록 높음 | "시키는 대로 움직이면 점수" |
| **angular_velocity_tracking** | +0.5 | 회전 속도 추적 정확도 | "회전도 정확하게" |
| **action_rate** | -0.01 | 행동이 급격히 변하면 감점 | "부드럽게 움직여야 함" |
| **joint_torque** | -0.0001 | 관절에 힘을 많이 쓰면 감점 | "에너지 아껴 쓰기" |

### 보상 공식 예시

```
총 보상 = (alive) + (upright) + (height) + (velocity_tracking) + ...
```

예를 들어:
- 똑바로 서서 멈춰있으면: 2.0 + 1.0 + 1.0 + 2.0 = **6.0**
- 약간 기울어지고 느리게 움직이면: 2.0 + 0.5 + 0.8 + 1.0 = **4.3**
- 쓰러지면: 에피소드 종료, 보상 없음

---

## ⚙️ 하이퍼파라미터 조정

### 학습 속도가 느릴 때

```powershell
# 학습률 높이기
python scripts/train.py --stage stand --lr 1e-3

# 롤아웃 스텝 늘리기
python scripts/train.py --stage stand --num-steps 48
```

### 학습이 불안정할 때

```powershell
# 학습률 낮추기
python scripts/train.py --stage stand --lr 1e-4

# 환경 수 줄이기 (배치 안정성)
python scripts/train.py --stage stand --num-envs 512
```

### GPU 온도가 높을 때 🌡️

```powershell
# 환경 수 줄이기 (GPU 부하 감소)
python scripts/train.py --stage stand --num-envs 512

# MuJoCo CPU 백엔드 사용 (느리지만 GPU 사용 안함)
python scripts/train.py --stage stand --backend mujoco --device cpu --num-envs 16
```

---

## 🔧 문제 해결

### "CUDA out of memory" 오류

```powershell
# 환경 수 줄이기
python scripts/train.py --stage stand --num-envs 256
```

### Genesis 설치 실패

```powershell
# pip 캐시 지우고 재설치
pip cache purge
pip install genesis-world --no-cache-dir
```

### "ModuleNotFoundError: No module named 'genesis'"

```powershell
# 가상환경 활성화 확인
.\venv\Scripts\Activate

# Genesis 재설치
pip install genesis-world
```

### MuJoCo 뷰어가 안 열림

Windows에서는 추가 설정이 필요할 수 있습니다:
```powershell
pip install glfw PyOpenGL PyOpenGL_accelerate
```

---

## 📊 학습 모니터링

### TensorBoard 사용

```powershell
# TensorBoard 실행
tensorboard --logdir=runs

# 브라우저에서 http://localhost:6006 접속
```

### progress.csv 분석

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('runs/microduck_stand_*/progress.csv')
df.plot(x='iteration', y='mean_reward', title='학습 진행')
plt.show()
```

---

## 🎓 추가 학습 자료

### 강화학습 기초
- [OpenAI Spinning Up](https://spinningup.openai.com/) - RL 기초 튜토리얼
- [Stable Baselines3 문서](https://stable-baselines3.readthedocs.io/)

### MuJoCo / Genesis
- [MuJoCo 공식 문서](https://mujoco.readthedocs.io/)
- [Genesis 공식 문서](https://genesis-world.readthedocs.io/)

### 관련 프로젝트
- [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) - 실제 로봇
- [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) - MuJoCo 환경

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다.

로봇 모델은 [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini)의 조인트 구조에서 영감을 받았으며, 해당 프로젝트는 Apache 2.0 라이선스입니다.
