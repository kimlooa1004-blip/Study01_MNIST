# Study01_MNIST — 손글씨 숫자 인식

**학번 2601922 · 이름 김루아**

MNIST로 학습한 CNN으로 손으로 쓴 숫자(0~9)를 인식합니다.
같은 모델을 **웹 페이지**와 **데스크톱 앱** 두 가지로 만들었습니다.

- 깃허브 저장소: https://github.com/kimlooa1004-blip/Study01_MNIST
- 깃허브 페이지: https://kimlooa1004-blip.github.io/Study01_MNIST/

## 웹 버전 (`web_version/`)

브라우저에서 마우스나 손가락으로 숫자를 그리면 바로 인식합니다.
깃허브 페이지는 정적 파일만 제공하므로, 학습된 가중치를 파일로 내보내
**자바스크립트가 직접 합성곱 계산을 수행합니다.** 외부 라이브러리를 쓰지 않습니다.

로컬에서 확인하려면:

```bash
python -m http.server 8000 --directory web_version
```

## 데스크톱 버전 (`desktop_version/`)

PyTorch + tkinter로 만든 창 프로그램입니다.

```bash
cd desktop_version
python train.py    # 학습 → mnist_cnn.pt 저장
python app.py      # 앱 실행
```

## 모델

```
입력 1x28x28
  → 합성곱(1→32) → ReLU → 최대풀링   (28→14)
  → 합성곱(32→64) → ReLU → 최대풀링  (14→7)
  → 완전연결(3136→128) → ReLU
  → 완전연결(128→10)
```

- 학습: Adam, 학습률 1e-3, 5에폭
- **MNIST 테스트 정확도 99.08%**
- 웹 버전의 자바스크립트 추론은 PyTorch와 확률 오차 `1.8e-7` 이내로 일치합니다.

## 전처리

캔버스를 그대로 줄이면 인식률이 떨어집니다. MNIST가 숫자를 20x20 상자에 맞춘 뒤
무게중심을 가운데 놓고 만든 자료이기 때문에, 같은 순서로 맞춥니다.

1. 글씨가 있는 영역만 잘라내기
2. 비율을 지키며 긴 변을 20px로 줄이기
3. 28x28 한가운데에 붙이기
4. 픽셀 무게중심을 정중앙으로 옮기기
5. `(값/255 - 0.1307) / 0.3081`로 정규화

## 폴더 구조

```
Study01_MNIST/
├── index.html              웹 앱으로 보내는 안내 페이지
├── CLAUDE.md               저장소 전체 지침
├── CLAUDE_전역.md          전역 CLAUDE.md 사본
├── desktop_version/
│   ├── CLAUDE.md
│   ├── model.py            CNN 구조
│   ├── train.py            학습
│   ├── recognize_gui.py    그림판 + 인식
│   ├── app.py              실행 진입점
│   └── mnist_cnn.pt        학습된 가중치
└── web_version/
    ├── CLAUDE.md
    ├── index.html
    ├── style.css
    ├── cnn.js              자바스크립트 CNN 추론
    ├── app.js              그리기 + 전처리
    ├── export_weights.py   가중치 내보내기
    └── weights/            브라우저용 가중치
```
