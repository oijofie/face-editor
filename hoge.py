from utils import ImageProcessor, FaceDetector, FaceTracker, VideoProcessor, UIHelper
import os
import cv2
import numpy as np
import tqdm
import argparse
import tkinter as tk
from tkinter import filedialog


def face_tracker_app(
    video_path, overlay_path, scene_change_threshold=30.0, output_path=None
):
    """
    カット検出に対応した顔トラッキングアプリケーション

    Parameters:
    video_path (str): 処理する動画のパス
    overlay_path (str): オーバーレイする画像のパス
    scene_change_threshold (float): シーン変更を検出する閾値
    output_path (str, optional): 出力ビデオのパス

    Returns:
    str: 出力動画のパス
    """
    # 各コンポーネントのインスタンス化
    image_processor = ImageProcessor()
    face_detector = FaceDetector()
    face_tracker = FaceTracker(tracker_type="CSRT")
    video_processor = VideoProcessor(video_path, output_path)
    ui_helper = UIHelper()

    # ファイルの存在確認
    if not os.path.isfile(video_path) or not os.path.isfile(overlay_path):
        print("エラー: ファイルが見つかりません。")
        return None

    # オーバーレイ画像の読み込み
    overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay_img is None:
        print("エラー: オーバーレイ画像を読み込めませんでした。")
        return None

    # 画像にアルファチャンネルがない場合は追加
    if overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)

    # ビデオを開く
    if not video_processor.open_video():
        print("エラー: ビデオを開けませんでした。")
        return None

    # ビデオライターを作成
    video_processor.create_writer()

    # 最初のフレームを読み込む
    ret, frame = video_processor.read_frame()
    if not ret:
        print("エラー: ビデオからフレームを読み込めませんでした。")
        video_processor.release()
        return None

    # 処理時間を短縮するための縮小係数の計算
    target_width = min(640, video_processor.frame_width)
    scale_factor = target_width / video_processor.frame_width

    # 顔検出
    faces = face_detector.detect_faces(frame, scale_factor)

    # 顔が検出されなかった場合
    if len(faces) == 0:
        print("最初のフレームで顔が検出されませんでした。")
        video_processor.release()
        return None

    # 検出された顔を表示して選択
    face_choice = ui_helper.display_faces(frame, faces)

    # 選択された顔の座標を取得
    x, y, w, h = faces[face_choice]
    bbox = (x, y, w, h)

    # トラッカーを初期化
    face_tracker.init_tracker(frame, bbox)

    # 顔の再検出用カウンター
    frame_count = 0
    redetection_interval = 30  # 30フレームごとに顔を再検出

    # ビデオの先頭に戻る
    video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 進捗表示用のtqdmバー
    pbar = tqdm.tqdm(total=video_processor.total_frames, desc="処理中")

    # 各フレームを処理
    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break

        frame_count += 1
        pbar.update(1)  # 進捗バーを更新

        # トラッカーを更新
        track_success, bbox = face_tracker.update_tracker(frame)

        # 定期的またはトラッキングが失われた場合またはシーン変更時に顔を再検出
        if frame_count % redetection_interval == 0 or not track_success:
            faces = face_detector.detect_faces(frame, scale_factor)  # 再度顔を検出

            # 顔が複数検出された場合は最も一致するものを選択。ない場合は画像をオーバーレイしない
            if len(faces) > 0:
                best_match_idx = 0

                # シーン変更時またはトラッキングが失敗した場合
                if not track_success:
                    # 顔検出結果を表示して選択
                    best_match_idx = ui_helper.display_faces(frame, faces)

                # 選択された顔で更新
                x, y, w, h = faces[best_match_idx]
                bbox = (x, y, w, h)

                # カット検出時や追跡失敗時はトラッカーをリセット
                if not track_success:
                    face_tracker.reset_tracker(frame, bbox)
                    track_success = True

        if track_success:
            # トラッキング成功
            x, y, w, h = [int(v) for v in bbox]

            # トラッキングされた顔に画像をオーバーレイ
            frame = image_processor.overlay_image(frame, overlay_img, x, y, w, h)

        # フレームを出力ビデオに書き込む
        video_processor.write_frame(frame)

    # リソースを解放
    video_processor.release()
    pbar.close()

    # 処理完了メッセージ
    print(f"\n処理完了。出力は {video_processor.output_path} に保存されました。")

    return video_processor.output_path


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
        default=30.0,
        help="カット検出の閾値（デフォルト: 30.0）",
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
