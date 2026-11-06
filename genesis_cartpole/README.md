# Genesis CartPole

Gymnasium의 `CartPole-v1`을 가져오지 않고, Genesis 안에 카트와 막대 장면을 직접 만든 입문용 환경입니다. Microduck 보행 학습과는 별개이고, "환경을 직접 작성한다"가 어떤 코드인지 보기 위한 작은 예제입니다.

## 장면

- 레일 위를 미끄러지는 카트 (질량 1kg, x축 슬라이드 조인트)
- 카트에 힌지로 연결된 막대 (질량 0.1kg, 길이 1m 캡슐)
- 중력 9.8 m/s², 제어 주기 0.02초
- 행동 `0`은 카트를 왼쪽(-x)으로 10N, `1`은 오른쪽(+x)으로 10N

호출 형태는 Gym과 비슷하지만 `reset`은 관측만 돌려줍니다.

```python
obs = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

관측 4개는 카트 위치, 카트 속도, 막대 각도, 막대 각속도입니다. 각도 0은 막대가 위를 향할 때이고, 부호는 +Y 힌지의 오른손 법칙을 따릅니다.

보상은 에피소드가 끝나기 전까지 매 스텝 `+1`입니다. 실패로 끝난 그 스텝도 `+1`이라서, 에피소드 반환값은 버틴 스텝 수와 같습니다.

종료는 두 가지로 나뉩니다.

- `terminated`: 막대 각도의 절댓값이 12도(약 0.209 rad)를 넘거나, 카트 위치의 절댓값이 2.4m를 넘음
- `truncated`: 실패 없이 500스텝에 도달

물리 엔진은 이 한계에서 카트를 붙잡지 않습니다. 슬라이드 조인트에는 범위 제한이 없고, 12도와 2.4m는 `genesis_cartpole/env.py`가 스텝 이후에 검사합니다.

## Gymnasium CartPole-v1과 다른 점

같은 규칙으로 맞춘 것:

- 카트 1kg, 막대 0.1kg, 막대 길이 1m, 힘 ±10N, 중력 9.8, 제어 주기 0.02초
- 초기 상태 네 성분을 [-0.05, 0.05]에서 균등하게 뽑음
- 관측 순서, +1 보상, 12도 / ±2.4m / 500스텝 종료

다른 것:

- Gymnasium은 균일 막대 공식을 세미 암시적 오일러로 한 번 적분합니다. 이 환경은 `assets/cartpole.xml`을 Genesis 강체 솔버가 적분합니다. 같은 행동을 넣어도 궤적은 같지 않습니다.
- 막대는 반지름 2cm 캡슐입니다. 질량 중심은 힌지에서 0.5m이고, 관성은 캡슐 형상에서 계산됩니다.
- 레일 지옴은 보이기만 합니다. 충돌로 막대를 세우지 않습니다.
- `reset()`은 관측만 반환합니다. 최신 Gymnasium은 `(obs, info)`를 반환합니다.
- 환경은 하나뿐입니다. Microduck 훈련처럼 병렬로 수천 개를 돌리지 않습니다.

## 설치

저장소 루트의 `requirements.txt`에 torch와 `genesis-world`가 이미 적혀 있습니다. 이 예제를 CPU에서 돌리려면 아래가 확인된 조합입니다.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "genesis-world==1.4.5"
```

`genesis-world`는 torch를 자동으로 설치하지 않습니다. torch 없이 `import genesis`를 하면 설치 안내와 함께 실패합니다.

헤드리스 Linux에서는 뷰어를 꺼도 `scene.build()`가 오프스크린 OpenGL을 만듭니다. EGL이 없으면 빌드가 실패합니다.

```bash
sudo apt-get install -y libegl1 libgl1-mesa-dri
```

GPU가 없는 머신에서 드라이버가 소프트웨어 렌더를 고르지 못하면 앞에 `LIBGL_ALWAYS_SOFTWARE=1`을 붙입니다. NVIDIA GPU가 있으면 `--` 옵션 없이 같은 명령을 쓰면 되고, 환경은 기본으로 `gs.cpu`를 사용합니다. GPU 백엔드는 `CartPoleEnv(backend="gpu")`로만 바꿉니다.

## 실행

저장소 루트에서 실행합니다.

```bash
# 화면 없이 무작위 행동 3에피소드
python -m genesis_cartpole.rollout --episodes 3

# 화면이 있을 때 뷰어
python -m genesis_cartpole.rollout --episodes 1 --render

# REINFORCE 200에피소드 (CPU, 창 없음)
python -m genesis_cartpole.train_reinforce --episodes 200

# 학습하는 모습을 창으로 보기. --speed 2는 실시간의 두 배.
# --play는 학습이 끝난 정책을 그 창에서 반복 재생한다.
python -m genesis_cartpole.train_reinforce --episodes 200 --render --speed 2 --play

# 보상과 종료 검사. 창을 열지 않는다.
python -m genesis_cartpole.test_cartpole
```

학습 로그의 `return`은 그 에피소드에서 받은 보상의 합, 즉 버틴 스텝 수입니다. 500에 가까워지면 시간 제한까지 막대를 세운 것입니다.
