// -*- coding: utf-8 -*-
/**
 * 브라우저에서 동작하는 CNN 추론 코드
 *
 * 데스크톱 버전(desktop_version/model.py)의 CNN모델과 똑같은 계산을 자바스크립트로 옮긴 것이다.
 * 구조: 합성곱(1→32) → ReLU → 최대풀링 → 합성곱(32→64) → ReLU → 최대풀링
 *       → 펼치기(3136) → 완전연결(128) → ReLU → 완전연결(10)
 *
 * 드롭아웃은 학습할 때만 쓰는 층이라 추론에서는 계산하지 않는다.
 */

/** 가중치 파일(.bin)과 설명 파일(.json)을 읽어 층별 배열로 나눠 담는다 */
async function 가중치_불러오기(기본경로) {
  const [설명응답, 이진응답] = await Promise.all([
    fetch(`${기본경로}/mnist_cnn.json`),
    fetch(`${기본경로}/mnist_cnn.bin`),
  ]);
  if (!설명응답.ok) throw new Error(`설명 파일을 읽지 못했습니다 (${설명응답.status})`);
  if (!이진응답.ok) throw new Error(`가중치 파일을 읽지 못했습니다 (${이진응답.status})`);

  const 설명 = await 설명응답.json();
  const 버퍼 = await 이진응답.arrayBuffer();
  const 전체 = new Float32Array(버퍼);

  if (전체.length !== 설명.전체개수) {
    throw new Error(`가중치 개수가 맞지 않습니다 (${전체.length} ≠ ${설명.전체개수})`);
  }

  const 가중치 = {};
  for (const 항목 of 설명.층) {
    가중치[항목.이름] = 전체.subarray(항목.시작, 항목.시작 + 항목.개수);
    가중치[항목.이름 + ".모양"] = 항목.모양;
  }
  return 가중치;
}

/**
 * 3x3 합성곱 (padding=1, stride=1) — 입력과 출력의 가로세로 크기가 같다
 * @param {Float32Array} 입력 - [입력채널, 크기, 크기] 순서로 펼쳐진 배열
 */
function 합성곱3x3(입력, 입력채널수, 크기, 커널, 편향, 출력채널수) {
  const 출력 = new Float32Array(출력채널수 * 크기 * 크기);

  for (let 출채널 = 0; 출채널 < 출력채널수; 출채널++) {
    const 출력시작 = 출채널 * 크기 * 크기;
    const 편향값 = 편향[출채널];

    for (let y = 0; y < 크기; y++) {
      for (let x = 0; x < 크기; x++) {
        let 합 = 편향값;

        for (let 입채널 = 0; 입채널 < 입력채널수; 입채널++) {
          const 입력시작 = 입채널 * 크기 * 크기;
          // 커널 배열 모양: [출력채널, 입력채널, 3, 3]
          const 커널시작 = ((출채널 * 입력채널수) + 입채널) * 9;

          for (let 커y = 0; 커y < 3; 커y++) {
            const 원본y = y + 커y - 1;          // padding=1 이므로 -1 만큼 밀린다
            if (원본y < 0 || 원본y >= 크기) continue;

            for (let 커x = 0; 커x < 3; 커x++) {
              const 원본x = x + 커x - 1;
              if (원본x < 0 || 원본x >= 크기) continue;
              합 += 입력[입력시작 + 원본y * 크기 + 원본x] * 커널[커널시작 + 커y * 3 + 커x];
            }
          }
        }
        출력[출력시작 + y * 크기 + x] = 합;
      }
    }
  }
  return 출력;
}

/** 음수를 0으로 바꾼다 (ReLU) */
function 렐루(배열) {
  for (let i = 0; i < 배열.length; i++) {
    if (배열[i] < 0) 배열[i] = 0;
  }
  return 배열;
}

/** 2x2 최대 풀링 — 가로세로가 절반으로 줄어든다 */
function 최대풀링2x2(입력, 채널수, 크기) {
  const 새크기 = Math.floor(크기 / 2);
  const 출력 = new Float32Array(채널수 * 새크기 * 새크기);

  for (let 채널 = 0; 채널 < 채널수; 채널++) {
    const 입력시작 = 채널 * 크기 * 크기;
    const 출력시작 = 채널 * 새크기 * 새크기;

    for (let y = 0; y < 새크기; y++) {
      for (let x = 0; x < 새크기; x++) {
        const a = 입력[입력시작 + (y * 2) * 크기 + (x * 2)];
        const b = 입력[입력시작 + (y * 2) * 크기 + (x * 2 + 1)];
        const c = 입력[입력시작 + (y * 2 + 1) * 크기 + (x * 2)];
        const d = 입력[입력시작 + (y * 2 + 1) * 크기 + (x * 2 + 1)];
        출력[출력시작 + y * 새크기 + x] = Math.max(a, b, c, d);
      }
    }
  }
  return 출력;
}

/** 완전연결 층 — 가중치 모양은 [출력크기, 입력크기] */
function 완전연결(입력, 가중치, 편향, 출력크기) {
  const 입력크기 = 입력.length;
  const 출력 = new Float32Array(출력크기);

  for (let 출 = 0; 출 < 출력크기; 출++) {
    let 합 = 편향[출];
    const 행시작 = 출 * 입력크기;
    for (let 입 = 0; 입 < 입력크기; 입++) {
      합 += 입력[입] * 가중치[행시작 + 입];
    }
    출력[출] = 합;
  }
  return 출력;
}

/** 로짓 값을 0~1 확률로 바꾼다 (소프트맥스) */
function 소프트맥스(로짓) {
  const 최대값 = Math.max(...로짓);   // 큰 수 계산으로 인한 오버플로 방지
  const 결과 = new Float32Array(로짓.length);
  let 합 = 0;
  for (let i = 0; i < 로짓.length; i++) {
    결과[i] = Math.exp(로짓[i] - 최대값);
    합 += 결과[i];
  }
  for (let i = 0; i < 결과.length; i++) 결과[i] /= 합;
  return 결과;
}

/**
 * 28x28 정규화된 입력(길이 784)을 받아 숫자 0~9의 확률을 돌려준다
 * @returns {Float32Array} 길이 10의 확률 배열
 */
function 예측하기(입력784, 가중치) {
  // 1층: 합성곱 1→32, 28x28 유지 → ReLU → 풀링으로 14x14
  let 특징 = 합성곱3x3(입력784, 1, 28, 가중치["conv1.weight"], 가중치["conv1.bias"], 32);
  렐루(특징);
  특징 = 최대풀링2x2(특징, 32, 28);

  // 2층: 합성곱 32→64, 14x14 유지 → ReLU → 풀링으로 7x7
  특징 = 합성곱3x3(특징, 32, 14, 가중치["conv2.weight"], 가중치["conv2.bias"], 64);
  렐루(특징);
  특징 = 최대풀링2x2(특징, 64, 14);

  // 펼치기: 64채널 × 7 × 7 = 3136
  특징 = 완전연결(특징, 가중치["fc1.weight"], 가중치["fc1.bias"], 128);
  렐루(특징);

  const 로짓 = 완전연결(특징, 가중치["fc2.weight"], 가중치["fc2.bias"], 10);
  return 소프트맥스(로짓);
}
