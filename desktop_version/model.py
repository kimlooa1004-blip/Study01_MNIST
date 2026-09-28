# -*- coding: utf-8 -*-
"""
MNIST 숫자 분류용 CNN 모델 정의

학습 스크립트(train.py)와 인식 GUI(recognize_gui.py)에서 공통으로 사용한다.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN모델(nn.Module):
    """MNIST 숫자(0~9) 분류를 위한 간단한 CNN 모델"""

    def __init__(self):
        super().__init__()
        # 첫 번째 합성곱 층: 흑백 이미지(1채널) -> 32채널
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 두 번째 합성곱 층: 32채널 -> 64채널
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 과적합 방지를 위한 드롭아웃
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 완전연결(선형) 층: 28x28 이미지가 두 번의 풀링을 거쳐 7x7이 됨
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 출력: 0~9, 10개 클래스

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)          # 28x28 -> 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)          # 14x14 -> 7x7
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x  # 소프트맥스 이전의 로짓(logit) 값을 반환
