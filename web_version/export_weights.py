# -*- coding: utf-8 -*-
"""학습된 PyTorch 가중치(mnist_cnn.pt)를 브라우저용 파일로 내보내는 스크립트

깃허브 페이지는 정적 파일만 제공하므로 파이썬을 실행할 수 없다.
그래서 가중치를 float32 바이너리 한 덩어리(mnist_cnn.bin)로 저장하고,
각 층의 이름·모양·위치를 적은 설명 파일(mnist_cnn.json)을 함께 만든다.
자바스크립트(cnn.js)가 이 두 파일을 읽어 직접 합성곱 계산을 수행한다.

사용법:  python export_weights.py
"""

import json
import os
import sys

import torch

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")

스크립트폴더 = os.path.dirname(os.path.abspath(__file__))
가중치_파일 = os.path.join(스크립트폴더, "..", "desktop_version", "mnist_cnn.pt")
출력폴더 = os.path.join(스크립트폴더, "weights")

# 내보낼 순서 (cnn.js가 이 순서대로 읽는다)
층_순서 = [
    "conv1.weight", "conv1.bias",
    "conv2.weight", "conv2.bias",
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
]


def main():
    os.makedirs(출력폴더, exist_ok=True)

    가중치 = torch.load(가중치_파일, map_location="cpu")
    설명 = []
    조각들 = []
    현재위치 = 0  # 바이너리 안에서 몇 번째 float부터 시작하는지

    for 이름 in 층_순서:
        텐서 = 가중치[이름].contiguous().float()
        개수 = 텐서.numel()
        설명.append({
            "이름": 이름,
            "모양": list(텐서.shape),
            "시작": 현재위치,
            "개수": 개수,
        })
        조각들.append(텐서.reshape(-1))
        현재위치 += 개수

    전체 = torch.cat(조각들).numpy().astype("<f4")  # 리틀엔디언 float32
    바이너리_경로 = os.path.join(출력폴더, "mnist_cnn.bin")
    설명_경로 = os.path.join(출력폴더, "mnist_cnn.json")

    전체.tofile(바이너리_경로)
    with open(설명_경로, "w", encoding="utf-8") as 파일:
        json.dump({"전체개수": 현재위치, "층": 설명}, 파일, ensure_ascii=False, indent=2)

    print(f"내보내기 완료: {바이너리_경로} ({os.path.getsize(바이너리_경로) / 1024:.0f} KB)")
    print(f"설명 파일    : {설명_경로}")
    for 항목 in 설명:
        print(f"  {항목['이름']:<14} 모양={항목['모양']} 개수={항목['개수']}")


if __name__ == "__main__":
    main()
