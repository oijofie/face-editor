from utils import ImageProcessor, FaceDetector, FaceTracker, VideoProcessor, UIHelper
import os
import cv2
import numpy as np
import tqdm
import argparse
import tkinter as tk
from tkinter import filedialog


class SceneChangeDetector:
    """
    連続するフレーム間の類似度を分析し、シーンの切り替わりを検出するクラス
    """

    def __init__(self, threshold):
        """
        初期化

        Args:
            threshold (float): シーン変更を検出する閾値（値が低いほど敏感に反応）
        """
        self.threshold = threshold
        self.prev_frame_gray = None

    def detect(self, frame):
        """
        フレームを分析し、シーンの変化を検出する

        Args:
            frame: 現在のフレーム（BGR形式）

        Returns:
            bool: シーンの変化が検出された場合はTrue、それ以外はFalse
        """
        # グレースケールに変換
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 最初のフレームの場合
        if self.prev_frame_gray is None:
            self.prev_frame_gray = current_frame_gray
            return False

        # フレーム間の差分を計算
        frame_diff = cv2.absdiff(current_frame_gray, self.prev_frame_gray)

        # 差分画像の平均値を計算
        diff_mean = cv2.mean(frame_diff)[0]

        # 前のフレームを更新
        self.prev_frame_gray = current_frame_gray

        # 閾値を超えた場合はシーン変更と判定(True or Falseで返る)
        return diff_mean > self.threshold

    def reset(self):
        """状態をリセット"""
        self.prev_frame_gray = None


class FaceTrackerApp:
    def __init__(
        self, video_path, overlay_path, scene_change_threshold, output_path=None
    ):
        """
        video_path (str): 入力ビデオファイルのパス
        overlay_path (str): オーバーレイする画像ファイルのパス
        scene_change_threshold (float): 顔検知する閾値
        output_path (str): 出力ビデオファイルのパス
        """

        self.video_path = video_path
        self.overlay_path = overlay_path
        self.scene_change_threshold = scene_change_threshold
        self.output_path = output_path

        # 追跡用パラメータ
        self.frame_count = 0
        # self.redetection_interval = 30  # 30フレームごとに顔を再検出
        self.redetection_interval = 2000  # 30フレームごとに顔を再検出
        self.bbox = None
        self.scale_factor = None

        # オーバーレイ実行フラグ
        self.skip_overlay = False

        # 各コンポーネントのインスタンス化
        self.image_processor = ImageProcessor()
        self.face_detector = FaceDetector()
        self.face_tracker = FaceTracker(tracker_type="CSRT")
        self.video_processor = VideoProcessor(video_path, output_path)
        self.ui_helper = UIHelper()
        self.scene_detector = SceneChangeDetector(scene_change_threshold)

        # リソース
        self.overlay_img = None

    def _load_resources(self):
        """ファイルの読み見とチェック"""
        if not os.path.isfile(self.video_path) or not os.path.isfile(self.overlay_path):
            print("エラー: ファイルが見つかりません。")
            return False

        # オーバーレイ画像の読み込み
        self.overlay_img = cv2.imread(self.overlay_path, cv2.IMREAD_UNCHANGED)
        if self.overlay_img is None:
            print("エラー: オーバーレイ画像を読み込めませんでした。")
            return False

        # 画像にアルファチャンネルがない場合は追加
        if len(self.overlay_img.shape) == 2:  # グレースケール画像
            # グレースケールをBGRAに変換
            self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGRA)
        elif self.overlay_img.shape[2] == 3:  # BGRのみ（アルファなし）
            # BGRをBGRAに変換
            self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGR2BGRA)
            # アルファチャンネルを不透明（255）に設定
            self.overlay_img[:, :, 3] = 255
        elif self.overlay_img.shape[2] == 4:  # すでにBGRA
            # 何もしない（すでにアルファチャンネルがある）
            pass
        else:
            print(f"エラー: サポートされていない画像形式です: {self.overlay_img.shape}")
            return False

        # ビデオを開く
        if not self.video_processor.open_video():
            print("エラー: ビデオを開けませんでした。")
            return False

        # ビデオライターを作成
        self.video_processor.create_writer()

        return True

    def _initialize_tracking(self):
        """顔検出と追跡の初期化"""
        ret, frame = self.video_processor.read_frame()
        if not ret:
            print("エラー: ビデオからフレームを読み込めませんでした。")
            self.video_processor.release()
            return False

        # 処理時間を短縮するための縮小係数の計算
        target_width = min(640, self.video_processor.frame_width)
        self.scale_factor = target_width / self.video_processor.frame_width

        # 顔検出
        faces = self.face_detector.detect_faces(frame, self.scale_factor)

        # 顔が検出されなかった場合
        if len(faces) == 0:
            print("最初のフレームで顔が検出されませんでした。")
            self.video_processor.release()
            return False

        # 検出された顔を表示して選択
        face_choice = self.ui_helper.display_faces(frame, faces)

        # 選択された顔の座標を取得
        x, y, w, h = faces[face_choice]
        self.bbox = (x, y, w, h)

        # トラッカーを初期化
        self.face_tracker.init_tracker(frame, self.bbox)

        # ビデオの先頭に戻る
        self.video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return True

    def _redetect_face(self, frame, track_success, scene_change):
        """顔の再検出処理"""
        faces = self.face_detector.detect_faces(frame, self.scale_factor)
        print(f"再検出: 検出された顔の数: {len(faces)}")

        if len(faces) > 0:
            best_match_idx = 0

            # 顔検出結果を表示して選択
            best_match_idx = self.ui_helper.display_faces(frame, faces)
            print("best_match_idx", best_match_idx)

            # 9を選んだらoverlayしない
            if best_match_idx == 8:  # 9を選んだらスキップ
                print("オーバーレイをスキップする")
                self.skip_overlay = True
                return track_success
            else:
                # 選択された顔で更新
                self.skip_overlay = False
                x, y, w, h = faces[best_match_idx]
                self.bbox = (x, y, w, h)

                # トラッキング
                self.face_tracker.reset_tracker(frame, self.bbox)

        elif len(faces) == 0:
            print("顔が検出されなかったため、overlayしない")
            self.skip_overlay = True
            # 顔が今回も前のフレームでもされていなかったら、次のフレームで再検出を試みる
            # if self.bbox is None:
            #     print(f"self.bbox is None: {self.bbox}")
            #     return False  # 次のフレームで再検出を試みる
            # # 以前の顔が検出されていたら、トラッカーをリセット（シーンが変わったのに以前の顔の位置を捉えてしまうため）
            # else:
            #     print(f"self.bbox is not None: {self.bbox}")
            #     self.face_tracker.reset_tracker(frame, self.bbox)
            #     return False  # 次のフレームで再検出を試みる

        return True

    def _process_video(self):
        """ビデオフレームの処理"""
        # 進捗表示用のtqdmバー
        with tqdm.tqdm(total=self.video_processor.total_frames, desc="処理中") as pbar:
            # 各フレームを処理
            while True:
                ret, frame = self.video_processor.read_frame()
                if not ret:
                    break

                self.frame_count += 1
                pbar.update(1)  # 進捗バーを更新

                # トラッカーを更新
                track_success, self.bbox = self.face_tracker.update_tracker(frame)
                # シーン変更検出
                scene_change = self.scene_detector.detect(frame)
                if (
                    scene_change
                    or not track_success
                    or self.frame_count % self.redetection_interval == 0
                ):
                    seconds = self.frame_count / self.video_processor.fps
                    minutes = int(seconds // 60)
                    seconds = seconds % 60

                    if scene_change:
                        print(
                            f"時間 {minutes:02d}:{seconds:05.2f} (フレーム {self.frame_count}): シーン変更を検出しました。"
                        )
                    elif not track_success:
                        print(
                            f"時間 {minutes:02d}:{seconds:05.2f} (フレーム {self.frame_count}): 顔の追跡に失敗しました。"
                        )
                    elif self.frame_count % self.redetection_interval == 0:
                        print(
                            f"時間 {minutes:02d}:{seconds:05.2f} (フレーム {self.frame_count}): 定期的な顔の再検出を行います。"
                        )
                    track_success = self._redetect_face(
                        frame, track_success, scene_change
                    )

                if track_success:
                    # トラッキング成功
                    x, y, w, h = [int(v) for v in self.bbox]
                    # print(f"顔の座標: x={x}, y={y}, w={w}, h={h}")
                    if w > 0 and h > 0 and not self.skip_overlay:
                        # トラッキングされた顔に画像をオーバーレイ
                        frame = self.image_processor.overlay_image(
                            frame, self.overlay_img, x, y, w, h
                        )

                # フレームを出力ビデオに書き込む
                self.video_processor.write_frame(frame)
        # 処理完了メッセージ
        print(
            f"\n処理完了。出力は {self.video_processor.output_path} に保存されました。"
        )

        return self.video_processor.output_path

    def run(self):
        """実行"""
        try:
            if not self._load_resources():
                return None
            if not self._initialize_tracking():
                return None
            return self._process_video()
        finally:
            self.video_processor.release()


def face_tracker_app(
    video_path, overlay_path, scene_change_threshold, output_path=None
):
    print("face_tracker_app")
    app = FaceTrackerApp(video_path, overlay_path, scene_change_threshold, output_path)
    return app.run()


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="カット検出対応の顔トラッキングとオーバーレイアプリケーション"
    )
    parser.add_argument("--video", type=str, help="処理する動画ファイルのパス")
    parser.add_argument(
        "--overlay", type=str, help="オーバーレイする画像ファイルのパス"
    )
    parser.add_argument("--output", type=str, help="出力ファイルのパス（省略可）")
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,  # TODO: 最適化を探る。10にしてもシーン変更検出回数が変わらない
        help="カット検出の閾値（デフォルト: 30.0）低いほど反応に敏感",
    )

    args = parser.parse_args()

    # コマンドライン引数がない場合はファイル選択ダイアログを表示
    video_path = args.video
    overlay_path = args.overlay
    output_path = args.output

    if video_path is None or overlay_path is None:
        print("ファイル選択ダイアログを表示します...")
        root = tk.Tk()
        root.withdraw()

        if video_path is None:
            print("処理するビデオファイルを選択してください...")
            video_path = filedialog.askopenfilename(
                title="処理するビデオファイルを選択",
                filetypes=[
                    ("ビデオファイル", "*.mp4 *.avi *.mov *.wmv *.mkv"),
                    ("すべてのファイル", "*.*"),
                ],
            )

        if overlay_path is None and video_path:  # ビデオが選択された場合のみ
            print("オーバーレイする画像ファイルを選択してください...")
            overlay_path = filedialog.askopenfilename(
                title="オーバーレイする画像ファイルを選択",
                filetypes=[
                    ("画像ファイル", "*.png *.jpg *.jpeg *.bmp"),
                    ("PNGファイル(透過推奨)", "*.png"),
                    ("すべてのファイル", "*.*"),
                ],
            )

    # ファイルが選択されたかチェック
    if not video_path or not overlay_path:
        print("ファイル選択がキャンセルされたか、ファイルが選択されませんでした。")
        return

    print(f"処理するビデオ: {video_path}")
    print(f"オーバーレイする画像: {overlay_path}")

    # 処理を実行
    output_path = face_tracker_app(
        video_path,
        overlay_path,
        scene_change_threshold=args.threshold,
        output_path=output_path,
    )

    if output_path and os.path.exists(output_path):
        print(f"出力ファイル: {output_path}")

        # 出力ファイルを開くか確認
        open_file = input("出力ファイルを再生しますか？ (y/n): ").strip().lower()
        if open_file == "y":
            try:
                import subprocess
                import platform

                # OSに応じて適切なコマンドでファイルを開く
                if platform.system() == "Windows":
                    os.startfile(output_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(["open", output_path])
                else:  # Linux
                    subprocess.call(["xdg-open", output_path])
            except Exception as e:
                print(f"ファイルを開くことができませんでした: {e}")


if __name__ == "__main__":
    main()
