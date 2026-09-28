// -*- coding: utf-8 -*-
/**
 * 화면 동작 담당 코드 — 그림 그리기, 전처리, 결과 표시
 *
 * 전처리는 데스크톱 버전(desktop_version/recognize_gui.py)의 _전처리와 같은 순서로 한다.
 *   ① 글씨가 있는 영역만 잘라내기
 *   ② 가로세로 비율을 지키며 긴 변이 20픽셀이 되게 줄이기
 *   ③ 28x28 한가운데 붙이기
 *   ④ 픽셀 무게중심을 정중앙(13.5, 13.5)으로 옮기기
 *   ⑤ 학습 때와 같은 평균/표준편차로 정규화
 * MNIST가 이런 방식으로 만들어진 자료라서, 똑같이 맞춰야 인식이 잘 된다.
 */

const 붓굵기 = 18;
const 평균 = 0.1307;      // 학습 때 쓴 MNIST 평균
const 표준편차 = 0.3081;  // 학습 때 쓴 MNIST 표준편차

const 그림판 = document.getElementById("그림판");
const 그리기맥락 = 그림판.getContext("2d", { willReadFrequently: true });
const 미리보기 = document.getElementById("전처리미리보기");
const 미리보기맥락 = 미리보기.getContext("2d");
const 예측숫자칸 = document.getElementById("예측숫자");
const 확신도칸 = document.getElementById("확신도");
const 확률목록 = document.getElementById("확률목록");
const 안내문 = document.getElementById("안내문");
const 지우기단추 = document.getElementById("지우기단추");
const 인식단추 = document.getElementById("인식단추");

let 가중치 = null;
let 그리는중 = false;
let 무언가그렸다 = false;

// ---------------------------------------------------------------- 화면 준비

/** 확률 막대 10줄(숫자 0~9)을 미리 만들어 둔다 */
function 확률목록_만들기() {
  확률목록.innerHTML = "";
  for (let 숫자 = 0; 숫자 <= 9; 숫자++) {
    const 줄 = document.createElement("li");
    줄.className = "확률줄";
    줄.innerHTML = `<span class="숫자">${숫자}</span>` +
      `<span class="막대바탕"><span class="막대"></span></span>` +
      `<span class="값">0%</span>`;
    확률목록.appendChild(줄);
  }
}

/** 그림판을 검게 비운다 */
function 그림판_비우기() {
  그리기맥락.fillStyle = "#000";
  그리기맥락.fillRect(0, 0, 그림판.width, 그림판.height);
  그리기맥락.strokeStyle = "#fff";
  그리기맥락.lineWidth = 붓굵기;
  그리기맥락.lineCap = "round";
  그리기맥락.lineJoin = "round";
}

/** 결과 표시를 처음 상태로 되돌린다 */
function 결과_비우기() {
  예측숫자칸.textContent = "?";
  확신도칸.textContent = "";
  확률목록.querySelectorAll(".확률줄").forEach((줄) => {
    줄.classList.remove("으뜸");
    줄.querySelector(".막대").style.width = "0%";
    줄.querySelector(".값").textContent = "0%";
  });
  미리보기맥락.fillStyle = "#000";
  미리보기맥락.fillRect(0, 0, 28, 28);
}

// ---------------------------------------------------------------- 그리기

/** 마우스/손가락 위치를 그림판 안의 좌표로 바꾼다 (화면 크기가 줄어도 맞게) */
function 좌표구하기(이벤트) {
  const 상자 = 그림판.getBoundingClientRect();
  return {
    x: (이벤트.clientX - 상자.left) * (그림판.width / 상자.width),
    y: (이벤트.clientY - 상자.top) * (그림판.height / 상자.height),
  };
}

function 그리기시작(이벤트) {
  if (!가중치) return;
  그리는중 = true;
  그림판.setPointerCapture(이벤트.pointerId);
  const { x, y } = 좌표구하기(이벤트);
  그리기맥락.beginPath();
  그리기맥락.moveTo(x, y);
  // 점 하나만 찍어도 보이도록 짧은 선을 긋는다
  그리기맥락.lineTo(x + 0.01, y + 0.01);
  그리기맥락.stroke();
  무언가그렸다 = true;
}

function 그리는중처리(이벤트) {
  if (!그리는중) return;
  const { x, y } = 좌표구하기(이벤트);
  그리기맥락.lineTo(x, y);
  그리기맥락.stroke();
}

function 그리기끝(이벤트) {
  if (!그리는중) return;
  그리는중 = false;
  그림판.releasePointerCapture?.(이벤트.pointerId);
  인식실행();   // 손을 떼면 바로 인식
}

// ---------------------------------------------------------------- 전처리

/**
 * 그림판 내용을 모델 입력(길이 784 배열)으로 바꾼다.
 * 아무것도 그리지 않았으면 null을 돌려준다.
 */
function 전처리() {
  const 화면 = 그리기맥락.getImageData(0, 0, 그림판.width, 그림판.height);
  const 점들 = 화면.data;
  const 너비 = 그림판.width;
  const 높이 = 그림판.height;

  // ① 글씨가 있는 영역(밝기 > 0)의 외곽 상자 찾기
  let 왼쪽 = 너비, 오른쪽 = -1, 위 = 높이, 아래 = -1;
  for (let y = 0; y < 높이; y++) {
    for (let x = 0; x < 너비; x++) {
      if (점들[(y * 너비 + x) * 4] > 0) {   // 빨강 채널 = 밝기 (흰 글씨라서)
        if (x < 왼쪽) 왼쪽 = x;
        if (x > 오른쪽) 오른쪽 = x;
        if (y < 위) 위 = y;
        if (y > 아래) 아래 = y;
      }
    }
  }
  if (오른쪽 < 0) return null;   // 빈 그림판

  const 상자너비 = 오른쪽 - 왼쪽 + 1;
  const 상자높이 = 아래 - 위 + 1;

  // ② 긴 변이 20이 되도록 비율을 지켜 줄이고 ③ 28x28 가운데에 붙이기
  const 비율 = 20 / Math.max(상자너비, 상자높이);
  const 새너비 = Math.max(1, Math.round(상자너비 * 비율));
  const 새높이 = Math.max(1, Math.round(상자높이 * 비율));

  const 작은판 = document.createElement("canvas");
  작은판.width = 28;
  작은판.height = 28;
  const 작은맥락 = 작은판.getContext("2d", { willReadFrequently: true });
  작은맥락.fillStyle = "#000";
  작은맥락.fillRect(0, 0, 28, 28);
  작은맥락.imageSmoothingEnabled = true;
  작은맥락.imageSmoothingQuality = "high";
  작은맥락.drawImage(
    그림판,
    왼쪽, 위, 상자너비, 상자높이,                                  // 잘라낼 영역
    Math.floor((28 - 새너비) / 2), Math.floor((28 - 새높이) / 2),   // 붙일 위치
    새너비, 새높이
  );

  // ④ 픽셀 무게중심을 정중앙으로 옮기기
  const 작은자료 = 작은맥락.getImageData(0, 0, 28, 28).data;
  const 밝기 = new Float32Array(28 * 28);
  let 총합 = 0, 가중x = 0, 가중y = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const 값 = 작은자료[(y * 28 + x) * 4];
      밝기[y * 28 + x] = 값;
      총합 += 값;
      가중x += x * 값;
      가중y += y * 값;
    }
  }
  if (총합 === 0) return null;

  const 이동x = Math.round(13.5 - 가중x / 총합);
  const 이동y = Math.round(13.5 - 가중y / 총합);

  const 옮긴밝기 = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y++) {
    const 새y = y + 이동y;
    if (새y < 0 || 새y >= 28) continue;
    for (let x = 0; x < 28; x++) {
      const 새x = x + 이동x;
      if (새x < 0 || 새x >= 28) continue;
      옮긴밝기[새y * 28 + 새x] = 밝기[y * 28 + x];
    }
  }

  // 모델이 보는 그림을 화면에도 보여 준다
  const 미리보기자료 = 미리보기맥락.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const 값 = 옮긴밝기[i];
    미리보기자료.data[i * 4] = 값;
    미리보기자료.data[i * 4 + 1] = 값;
    미리보기자료.data[i * 4 + 2] = 값;
    미리보기자료.data[i * 4 + 3] = 255;
  }
  미리보기맥락.putImageData(미리보기자료, 0, 0);

  // ⑤ 학습 때와 같은 방식으로 정규화
  const 입력 = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    입력[i] = (옮긴밝기[i] / 255 - 평균) / 표준편차;
  }
  return 입력;
}

// ---------------------------------------------------------------- 인식

function 인식실행() {
  if (!가중치) return;

  const 입력 = 전처리();
  if (!입력) {
    안내문.textContent = "먼저 숫자를 그려 주세요.";
    결과_비우기();
    return;
  }

  const 시작시각 = performance.now();
  const 확률 = 예측하기(입력, 가중치);
  const 걸린시간 = performance.now() - 시작시각;

  let 으뜸숫자 = 0;
  for (let 숫자 = 1; 숫자 <= 9; 숫자++) {
    if (확률[숫자] > 확률[으뜸숫자]) 으뜸숫자 = 숫자;
  }

  예측숫자칸.textContent = String(으뜸숫자);
  확신도칸.textContent = `확신도 ${(확률[으뜸숫자] * 100).toFixed(1)}%`;

  확률목록.querySelectorAll(".확률줄").forEach((줄, 숫자) => {
    const 값 = 확률[숫자] * 100;
    줄.classList.toggle("으뜸", 숫자 === 으뜸숫자);
    줄.querySelector(".막대").style.width = `${값}%`;
    줄.querySelector(".값").textContent = `${값.toFixed(1)}%`;
  });

  안내문.textContent = `계산 시간 ${걸린시간.toFixed(1)}밀리초`;
}

// ---------------------------------------------------------------- 시작

async function 시작하기() {
  확률목록_만들기();
  그림판_비우기();
  결과_비우기();
  인식단추.disabled = true;

  그림판.addEventListener("pointerdown", 그리기시작);
  그림판.addEventListener("pointermove", 그리는중처리);
  그림판.addEventListener("pointerup", 그리기끝);
  그림판.addEventListener("pointercancel", 그리기끝);

  지우기단추.addEventListener("click", () => {
    그림판_비우기();
    결과_비우기();
    무언가그렸다 = false;
    안내문.textContent = "숫자를 그려 보세요.";
  });
  인식단추.addEventListener("click", 인식실행);

  try {
    가중치 = await 가중치_불러오기("weights");
    인식단추.disabled = false;
    안내문.textContent = "준비됐습니다. 숫자를 그려 보세요.";
  } catch (오류) {
    안내문.textContent = `모델을 불러오지 못했습니다: ${오류.message}`;
    console.error(오류);
  }
}

시작하기();
