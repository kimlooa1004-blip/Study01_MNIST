# Study01_MNIST — 손글씨 숫자 인식

MNIST 자료로 학습한 CNN으로 손글씨 숫자(0~9)를 인식하는 과제 저장소다.
같은 모델을 **데스크톱 앱**과 **웹 페이지** 두 가지 형태로 제공한다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `index.html` | 저장소 첫 주소로 들어왔을 때 `web_version/`으로 보내는 안내 페이지 |
| `desktop_version/` | PyTorch 학습 코드 + tkinter 데스크톱 앱 |
| `web_version/` | 깃허브 페이지로 배포되는 웹 앱 (자바스크립트 추론) |
| `CLAUDE_전역.md` | 전역 CLAUDE.md 사본 (과제 제출용) |
| `대화기록/` | 클로드와 주고받은 대화 기록 (과제 제출용) |

## 작성 규칙

- **모든 코드, 주석, 변수명, 함수명을 한글로 작성한다.** 기존 파일의 방식을 그대로 따른다.
- 파일은 UTF-8로 저장한다. PowerShell 스크립트를 만들 때는 BOM을 붙여야 한글이 깨지지 않는다.
- 콘솔 출력 전에 `sys.stdout`이 `None`인지 확인한다. `pythonw.exe`로 실행하면 표준출력이 없다.

## 두 버전이 반드시 같아야 하는 부분

데스크톱과 웹이 **같은 그림에 같은 답**을 내야 한다. 아래를 한쪽만 고치면 안 된다.

1. **모델 구조** — `desktop_version/model.py`의 `CNN모델`과 `web_version/cnn.js`의 `예측하기`
2. **전처리 순서** — `recognize_gui.py`의 `_전처리`와 `web_version/app.js`의 `전처리`
   (외곽 잘라내기 → 긴 변 20px로 축소 → 28x28 가운데 배치 → 무게중심 이동 → 정규화)
3. **정규화 상수** — 평균 `0.1307`, 표준편차 `0.3081` (MNIST 학습 때 쓴 값)

모델을 다시 학습시켰다면 `web_version/export_weights.py`를 실행해 웹용 가중치도 다시 내보내야 한다.

## 자주 쓰는 명령

```bash
# 모델 학습 (가중치를 desktop_version/mnist_cnn.pt로 저장)
cd desktop_version && python train.py

# 데스크톱 앱 실행
cd desktop_version && python app.py

# 웹용 가중치 다시 내보내기
cd web_version && python export_weights.py

# 웹 페이지를 로컬에서 확인 (file:// 로 열면 가중치 fetch가 막힌다)
python -m http.server 8000 --directory web_version
```

## 주의

- `desktop_version/data/`는 약 64MB라 깃에 올리지 않는다. `train.py`가 자동으로 내려받는다.
- `web_version/weights/mnist_cnn.bin`은 약 1.6MB이며 **깃에 포함한다.** 이 파일이 없으면 웹 앱이 동작하지 않는다.
