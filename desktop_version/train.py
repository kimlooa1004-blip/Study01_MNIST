# -*- coding: utf-8 -*-
"""
MNIST 손글씨 숫자 인식 모델 학습 스크립트

torchvision의 MNIST 데이터셋을 내려받아 간단한 CNN(합성곱 신경망)을 학습시키고,
학습이 끝난 가중치를 mnist_cnn.pt 파일로 저장한다.
"""

import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN모델

# Windows 콘솔에서 한글이 깨지지 않도록 표준출력을 UTF-8로 설정
# (pythonw.exe로 실행하면 콘솔이 없어 sys.stdout이 None이므로 먼저 확인한다)
if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")

# 어느 폴더에서 실행하든 이 파일이 있는 폴더를 기준으로 데이터/가중치 경로를 잡는다
스크립트폴더 = os.path.dirname(os.path.abspath(__file__))
데이터폴더 = os.path.join(스크립트폴더, "data")
가중치_파일 = os.path.join(스크립트폴더, "mnist_cnn.pt")

# 학습에 사용할 장치를 결정한다 (GPU가 있으면 GPU, 없으면 CPU)
장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def 데이터로더_만들기(배치크기=64):
    """MNIST 학습/테스트 데이터로더를 생성한다"""
    변환 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 평균/표준편차로 정규화
    ])

    학습데이터셋 = datasets.MNIST(root=데이터폴더, train=True, download=True, transform=변환)
    테스트데이터셋 = datasets.MNIST(root="./data", train=False, download=True, transform=변환)

    학습로더 = DataLoader(학습데이터셋, batch_size=배치크기, shuffle=True)
    테스트로더 = DataLoader(테스트데이터셋, batch_size=1000, shuffle=False)

    return 학습로더, 테스트로더


def 한_에폭_학습(모델, 로더, 옵티마이저, 에폭번호):
    """학습 데이터 한 바퀴(1 에폭)를 돌면서 모델을 학습시킨다"""
    모델.train()
    for 배치번호, (입력, 정답) in enumerate(로더):
        입력, 정답 = 입력.to(장치), 정답.to(장치)

        옵티마이저.zero_grad()
        출력 = 모델(입력)
        손실 = F.cross_entropy(출력, 정답)
        손실.backward()
        옵티마이저.step()

        if 배치번호 % 200 == 0:
            print(f"에폭 {에폭번호} [{배치번호 * len(입력)}/{len(로더.dataset)}] 손실값: {손실.item():.4f}")


def 모델_평가(모델, 로더):
    """테스트 데이터로 모델의 정확도를 평가한다"""
    모델.eval()
    총손실 = 0
    맞춘개수 = 0

    with torch.no_grad():
        for 입력, 정답 in 로더:
            입력, 정답 = 입력.to(장치), 정답.to(장치)
            출력 = 모델(입력)
            총손실 += F.cross_entropy(출력, 정답, reduction="sum").item()
            예측값 = 출력.argmax(dim=1)
            맞춘개수 += (예측값 == 정답).sum().item()

    평균손실 = 총손실 / len(로더.dataset)
    정확도 = 100.0 * 맞춘개수 / len(로더.dataset)
    print(f"테스트 결과 - 평균 손실: {평균손실:.4f}, 정확도: {정확도:.2f}%")
    return 정확도


def main():
    에폭수 = 5
    학습률 = 1e-3

    print(f"사용 장치: {장치}")
    학습로더, 테스트로더 = 데이터로더_만들기()

    모델 = CNN모델().to(장치)
    옵티마이저 = optim.Adam(모델.parameters(), lr=학습률)

    for 에폭 in range(1, 에폭수 + 1):
        한_에폭_학습(모델, 학습로더, 옵티마이저, 에폭)
        모델_평가(모델, 테스트로더)

    # 학습된 가중치를 파일로 저장
    torch.save(모델.state_dict(), 가중치_파일)
    print("학습 완료! 가중치를 mnist_cnn.pt 파일에 저장했습니다.")


if __name__ == "__main__":
    main()
