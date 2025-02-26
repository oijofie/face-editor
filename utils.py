import cv2
import numpy as np
import os
import argparse
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageProcessor:
    """画像処理に関するクラス"""

    @staticmethod
    def overlay_image(frame, overlay, x, y, w, h):
        """フレーム内の領域に画像をオーバーレイする（境界チェック付き）"""
        # 座標がフレーム内にあることを確認
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w > frame.shape[1]:
            w = frame.shape[1] - x
        if y + h > frame.shape[0]:
            h = frame.shape[0] - y

        # オーバーレイ画像を顔のサイズに合わせてリサイズ
        resized_overlay = cv2.resize(overlay, (w, h))

        # 実際のROI寸法を計算
        roi_h, roi_w = min(h, frame.shape[0] - y), min(w, frame.shape[1] - x)

        # 対象領域（ROI）を作成
        roi = frame[y : y + roi_h, x : x + roi_w]

        # 必要に応じてオーバーレイを調整
        resized_overlay = resized_overlay[:roi_h, :roi_w]

        # オーバーレイのマスクと逆マスクを作成
        overlay_alpha = resized_overlay[:, :, 3] / 255.0
        overlay_alpha = np.dstack([overlay_alpha] * 3)

        # オーバーレイ画像からBGRチャネルを抽出
        overlay_bgr = resized_overlay[:, :, :3]

        # 前景と背景を計算
        foreground = overlay_bgr * overlay_alpha
        background = roi * (1 - overlay_alpha)

        # 前景と背景を組み合わせる
        result = foreground + background

        # 結果をフレームに戻す
        frame[y : y + roi_h, x : x + roi_w] = result.astype(np.uint8)

        return frame


class FaceDetector:
    """顔検出に関するクラス"""

    def __init__(self):
        """顔検出器の初期化"""
        # haarcascade_frontalface_default.xml ファイルのパスを確認
        # OpenCVのインストールパスから見つける
        self.cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if not os.path.exists(self.cascade_path):
            # 標準パスにない場合、相対パスで試す
            self.cascade_path = "haarcascade_frontalface_default.xml"
            if not os.path.exists(self.cascade_path):
                print("警告: 顔検出用のカスケードファイルが見つかりません。")
                print(
                    "OpenCVのhaarcascade_frontalface_default.xmlをダウンロードしてください。"
                )

        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, frame, scale_factor=1.0):
        """
        フレーム内の顔を検出する

        Parameters:
        frame (numpy.ndarray): 顔を検出するフレーム
        scale_factor (float): 処理高速化のための縮小係数

        Returns:
        list: 検出された顔の座標リスト [(x, y, w, h), ...]
        """
        # 処理速度向上のためにフレームをリサイズ
        if scale_factor != 1.0:
            frame_height, frame_width = frame.shape[:2]
            target_width = int(frame_width * scale_factor)
            target_height = int(frame_height * scale_factor)
            small_frame = cv2.resize(frame, (target_width, target_height))
        else:
            small_frame = frame.copy()

        # グレースケールに変換
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # 顔検出
        faces_small = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        # 検出された顔の座標を元のサイズに戻す
        faces = []
        for x, y, w, h in faces_small:
            faces.append(
                (
                    int(x / scale_factor),
                    int(y / scale_factor),
                    int(w / scale_factor),
                    int(h / scale_factor),
                )
            )

        return faces


class FaceTracker:
    """顔追跡に関するクラス"""

    def __init__(self, tracker_type="CSRT"):
        """
        トラッカーの初期化

        Parameters:
        tracker_type (str): 使用するトラッカーのタイプ ('CSRT', 'KCF', 'MOSSE' など)
        """
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker(tracker_type)

    def _create_tracker(self, tracker_type):
        """指定されたタイプのトラッカーを作成"""
        # OpenCV 4.xで利用可能なトラッカーを確認
        if tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        elif tracker_type == "MOSSE":
            return cv2.TrackerMOSSE_create()
        else:
            # デフォルトではCSRTを使用
            return cv2.TrackerCSRT_create()

    def init_tracker(self, frame, bbox):
        """
        トラッカーを初期化

        Parameters:
        frame (numpy.ndarray): 初期化するフレーム
        bbox (tuple): 追跡する顔の座標 (x, y, w, h)

        Returns:
        bool: 初期化が成功したかどうか
        """
        return self.tracker.init(frame, bbox)

    def update_tracker(self, frame):
        """
        トラッカーを更新

        Parameters:
        frame (numpy.ndarray): 現在のフレーム

        Returns:
        tuple: (追跡成功フラグ, 追跡対象の座標)
        """
        return self.tracker.update(frame)

    def reset_tracker(self, frame, bbox):
        """
        トラッカーをリセットして再初期化

        Parameters:
        frame (numpy.ndarray): 現在のフレーム
        bbox (tuple): 新しい追跡対象の座標

        Returns:
        bool: 再初期化が成功したかどうか
        """
        self.tracker = self._create_tracker(self.tracker_type)
        return self.init_tracker(frame, bbox)


class VideoProcessor:
    """ビデオ処理に関するクラス"""

    def __init__(self, video_path, output_path=None):
        """
        ビデオプロセッサの初期化

        Parameters:
        video_path (str): 入力ビデオのパス
        output_path (str, optional): 出力ビデオのパス
        """
        self.video_path = video_path

        # 出力パスが指定されていない場合は自動生成
        if output_path is None:
            base_name = os.path.basename(video_path)
            name, ext = os.path.splitext(base_name)
            self.output_path = f"output_{name}{ext}"
        else:
            self.output_path = output_path

        # ビデオのプロパティ
        self.cap = None
        self.out = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.total_frames = 0

    def open_video(self):
        """ビデオを開き、プロパティを取得"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False

        # ビデオのプロパティを取得
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return True

    def create_writer(self, fourcc="mp4v"):
        """ビデオライターを作成"""
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.out = cv2.VideoWriter(
            self.output_path,
            fourcc_code,
            self.fps,
            (self.frame_width, self.frame_height),
        )
        return True

    def read_frame(self):
        """フレームを読み込む"""
        return self.cap.read()

    def write_frame(self, frame):
        """フレームを書き込む"""
        self.out.write(frame)

    def release(self):
        """リソースを解放"""
        if self.cap is not None:
            self.cap.release()

        if self.out is not None:
            self.out.release()


class UIHelper:
    """UI操作をサポートするヘルパークラス"""

    @staticmethod
    def display_faces(frame, faces):
        """検出された顔を表示し、インデックスを返す"""
        if len(faces) == 0:
            return 0

        # 顔に番号を振って表示
        display_frame = frame.copy()
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                display_frame,
                str(i + 1),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        # GUIで表示
        cv2.imshow(
            "select face to track (e.g. face 1 -> press '1')",
            display_frame,
        )

        # ユーザーから入力を受け取る
        face_idx = 0
        while True:
            key = cv2.waitKey(0) & 0xFF

            # 数字キーが押されたか確認
            if ord("1") <= key <= ord("9"):
                selected_idx = key - ord("1")  # 0から始まるインデックスに変換
                if selected_idx < len(faces):
                    face_idx = selected_idx
                    break

            # エスケープキーかEnterが押されたら最初の顔を選択
            if key == 27 or key == 13:  # ESCまたはEnter
                break

        cv2.destroyAllWindows()
        return face_idx
