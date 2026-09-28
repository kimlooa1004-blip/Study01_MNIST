# -*- coding: utf-8 -*-
"""
손글씨 숫자 인식 GUI

사용자가 캔버스에 마우스로 숫자를 그리면, 학습된 CNN 모델(mnist_cnn.pt)을 불러와서
어떤 숫자(0~9)인지 예측하여 화면에 보여준다.
"""

import os
import sys
import tkinter as tk

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model import CNN모델

# Windows 콘솔에서 한글이 깨지지 않도록 표준출력을 UTF-8로 설정
# (pythonw.exe로 실행하면 콘솔이 없어 sys.stdout이 None이므로 먼저 확인한다)
if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")

# 어느 폴더에서 실행하든 이 파일이 있는 폴더의 가중치를 쓴다
가중치_파일 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_cnn.pt")
캔버스_크기 = 280          # 화면에 보여줄 그림판 크기 (픽셀)
붓_굵기 = 18               # 그림을 그릴 때 선의 굵기
장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class 숫자인식앱:
    """마우스로 숫자를 그리고 예측 결과를 보여주는 tkinter 애플리케이션"""

    def __init__(self, 창):
        self.창 = 창
        self.창.title("손글씨 숫자 인식기")
        self.창.resizable(False, False)

        # 모델을 불러오고 추론(평가) 모드로 전환
        self.모델 = CNN모델().to(장치)
        self.모델.load_state_dict(torch.load(가중치_파일, map_location=장치))
        self.모델.eval()

        # 실제 그림은 흑백(L 모드) 이미지에 그려서 저장한다 (배경: 검은색, 글씨: 흰색)
        self.그림이미지 = Image.new("L", (캔버스_크기, 캔버스_크기), color=0)
        self.그림도구 = ImageDraw.Draw(self.그림이미지)

        self._위젯_구성()

        self.이전좌표 = None

    def _위젯_구성(self):
        # 숫자를 그리는 캔버스 (화면 표시용, 배경 검정 / 글씨 흰색)
        self.캔버스 = tk.Canvas(
            self.창, width=캔버스_크기, height=캔버스_크기, bg="black", cursor="cross"
        )
        self.캔버스.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.캔버스.bind("<B1-Motion>", self._그리는_중)
        self.캔버스.bind("<ButtonRelease-1>", self._그리기_종료)

        # 예측 결과를 보여주는 라벨
        self.결과라벨 = tk.Label(self.창, text="숫자를 그려보세요", font=("맑은 고딕", 18))
        self.결과라벨.grid(row=1, column=0, columnspan=2)

        # 상위 3개 후보를 보여주는 라벨
        self.후보라벨 = tk.Label(self.창, text="", font=("맑은 고딕", 11), fg="gray")
        self.후보라벨.grid(row=2, column=0, columnspan=2, pady=(0, 5))

        # 지우기 버튼
        지우기버튼 = tk.Button(self.창, text="지우기", command=self._지우기, width=12)
        지우기버튼.grid(row=3, column=0, pady=10)

        # 예측 버튼
        예측버튼 = tk.Button(self.창, text="예측하기", command=self._예측하기, width=12)
        예측버튼.grid(row=3, column=1, pady=10)

    def _그리는_중(self, 이벤트):
        x, y = 이벤트.x, 이벤트.y

        if self.이전좌표 is not None:
            이전x, 이전y = self.이전좌표
            # 화면 캔버스에 선을 그린다
            self.캔버스.create_line(
                이전x, 이전y, x, y, width=붓_굵기, fill="white",
                capstyle=tk.ROUND, smooth=True,
            )
            # 실제 예측에 사용할 이미지에도 같은 선을 그린다
            self.그림도구.line([이전x, 이전y, x, y], fill=255, width=붓_굵기)

        self.이전좌표 = (x, y)

    def _그리기_종료(self, 이벤트):
        self.이전좌표 = None
        # 붓을 뗄 때마다 자동으로 예측해서 결과를 갱신한다
        self._예측하기()

    def _지우기(self):
        self.캔버스.delete("all")
        self.그림이미지 = Image.new("L", (캔버스_크기, 캔버스_크기), color=0)
        self.그림도구 = ImageDraw.Draw(self.그림이미지)
        self.결과라벨.config(text="숫자를 그려보세요")
        self.후보라벨.config(text="")

    def _전처리(self):
        """그린 그림을 MNIST 데이터와 같은 형태(28x28, 중앙 정렬, 정규화)로 변환한다

        MNIST는 숫자를 20x20 상자에 맞춰 축소한 뒤, 픽셀 무게중심을 28x28 이미지의
        한가운데에 오도록 옮겨서 만들어졌다. 같은 방식으로 맞춰 줘야 인식률이 높다.
        반환값이 None이면 아무것도 그리지 않은 상태를 뜻한다.
        """
        원본 = np.array(self.그림이미지, dtype=np.float32)

        # 글씨가 있는 영역(0보다 큰 픽셀)의 외곽 상자를 찾는다
        칠해진_행 = np.where(원본.max(axis=1) > 0)[0]
        칠해진_열 = np.where(원본.max(axis=0) > 0)[0]
        if len(칠해진_행) == 0 or len(칠해진_열) == 0:
            return None  # 빈 캔버스

        위, 아래 = 칠해진_행[0], 칠해진_행[-1] + 1
        왼쪽, 오른쪽 = 칠해진_열[0], 칠해진_열[-1] + 1
        잘라낸이미지 = self.그림이미지.crop((왼쪽, 위, 오른쪽, 아래))

        # 가로세로 비율을 유지한 채 긴 변이 20픽셀이 되도록 축소한다
        폭, 높이 = 잘라낸이미지.size
        비율 = 20.0 / max(폭, 높이)
        새폭 = max(1, int(round(폭 * 비율)))
        새높이 = max(1, int(round(높이 * 비율)))
        축소이미지 = 잘라낸이미지.resize((새폭, 새높이), Image.LANCZOS)

        # 28x28 검은 배경 한가운데에 일단 붙인다
        도화지 = Image.new("L", (28, 28), color=0)
        도화지.paste(축소이미지, ((28 - 새폭) // 2, (28 - 새높이) // 2))

        # 픽셀 무게중심이 정중앙(13.5, 13.5)에 오도록 평행이동한다
        배열 = np.array(도화지, dtype=np.float32)
        총합 = 배열.sum()
        if 총합 > 0:
            y좌표, x좌표 = np.indices(배열.shape)
            무게중심x = (x좌표 * 배열).sum() / 총합
            무게중심y = (y좌표 * 배열).sum() / 총합
            이동x = int(round(13.5 - 무게중심x))
            이동y = int(round(13.5 - 무게중심y))
            배열 = np.roll(배열, (이동y, 이동x), axis=(0, 1))

        # 0~1 범위로 바꾼 뒤, 학습 때와 동일한 평균/표준편차로 정규화
        입력텐서 = torch.from_numpy(배열).reshape(1, 1, 28, 28) / 255.0
        입력텐서 = (입력텐서 - 0.1307) / 0.3081
        return 입력텐서.to(장치)

    def _예측하기(self):
        입력텐서 = self._전처리()
        if 입력텐서 is None:
            self.결과라벨.config(text="숫자를 그려보세요")
            self.후보라벨.config(text="")
            return

        with torch.no_grad():
            출력 = self.모델(입력텐서)
            확률분포 = F.softmax(출력, dim=1)[0]

        예측숫자 = int(확률분포.argmax().item())
        확신도 = float(확률분포[예측숫자].item()) * 100
        self.결과라벨.config(text=f"예측 결과: {예측숫자}  (확신도 {확신도:.1f}%)")

        # 확률이 높은 순서로 상위 3개 후보를 함께 보여준다
        상위확률, 상위숫자 = torch.topk(확률분포, 3)
        후보문구 = "   ".join(
            f"{int(숫자)}: {float(확률) * 100:.1f}%"
            for 확률, 숫자 in zip(상위확률, 상위숫자)
        )
        self.후보라벨.config(text=후보문구)


def main():
    if not os.path.exists(가중치_파일):
        print(f"'{가중치_파일}' 파일이 없습니다. 먼저 python train.py 를 실행해 주세요.")
        return

    창 = tk.Tk()
    숫자인식앱(창)
    창.mainloop()


if __name__ == "__main__":
    main()
