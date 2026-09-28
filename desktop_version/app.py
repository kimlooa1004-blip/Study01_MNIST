# -*- coding: utf-8 -*-
"""손글씨 숫자 인식 앱 실행 진입점

바탕화면 바로가기(pythonw.exe)로 실행되는 파일이다. 콘솔 창이 없기 때문에
오류가 나면 아무 일도 일어나지 않은 것처럼 보이므로, 모든 예외를 잡아서
메시지 창으로 보여 준다.
"""

import os
import sys
import tkinter as tk
import traceback
from tkinter import messagebox

# 어느 폴더에서 실행하든 이 파일이 있는 폴더를 기준으로 동작하게 한다
앱폴더 = os.path.dirname(os.path.abspath(__file__))
os.chdir(앱폴더)
sys.path.insert(0, 앱폴더)

아이콘_파일 = os.path.join(앱폴더, "icon.ico")
가중치_파일 = os.path.join(앱폴더, "mnist_cnn.pt")

# 작업 표시줄에서 파이썬이 아닌 이 앱으로 인식되도록 고유 ID를 지정한다
# (바로가기에도 같은 ID가 들어 있어야 작업 표시줄 고정이 제대로 동작한다)
앱_ID = "MNIST.HandwrittenDigitRecognizer.1"


def _작업표시줄_ID_설정():
    """윈도우에 이 프로세스의 고유 앱 ID를 알려 준다 (윈도우가 아니면 무시)"""
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(앱_ID)
    except Exception:
        pass  # 다른 운영체제이거나 지원하지 않으면 그냥 넘어간다


def main():
    _작업표시줄_ID_설정()

    if not os.path.exists(가중치_파일):
        임시창 = tk.Tk()
        임시창.withdraw()
        messagebox.showerror(
            "손글씨 숫자 인식기",
            "학습된 가중치 파일(mnist_cnn.pt)을 찾을 수 없습니다.\n\n"
            f"폴더: {앱폴더}\n\n"
            "명령 프롬프트에서 python train.py 를 먼저 실행해 주세요.",
        )
        return

    # 무거운 라이브러리는 여기서 불러온다 (가중치가 없을 때 괜히 기다리지 않도록)
    from recognize_gui import 숫자인식앱

    창 = tk.Tk()
    if os.path.exists(아이콘_파일):
        try:
            창.iconbitmap(default=아이콘_파일)  # 창과 작업 표시줄 아이콘
        except tk.TclError:
            pass
    숫자인식앱(창)
    창.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 콘솔이 없으므로 오류 내용을 메시지 창으로 보여 준다
        오류내용 = traceback.format_exc()
        try:
            임시창 = tk.Tk()
            임시창.withdraw()
            messagebox.showerror("손글씨 숫자 인식기 - 오류", 오류내용)
        except Exception:
            print(오류내용)
