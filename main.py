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

    # 選択された顔の特徴を保存（カット検出後の顔選択に使用）
    target_face_roi = frame[y : y + h, x : x + w]
    target_face_hist = None

    # HSV色空間でヒストグラムを計算
    if target_face_roi.size > 0:
        face_roi_hsv = cv2.cvtColor(target_face_roi, cv2.COLOR_BGR2HSV)
        target_face_hist = cv2.calcHist(
            [face_roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]
        )
        cv2.normalize(target_face_hist, target_face_hist, 0, 255, cv2.NORM_MINMAX)

    # トラッカーを初期化
    face_tracker.init_tracker(frame, bbox)

    # 顔の再検出用カウンター
    frame_count = 0
    redetection_interval = 30  # 30フレームごとに顔を再検出

    # シーン変更検出用の変数
    prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        # シーン変更検出
        is_scene_change = False
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # フレーム間の差分を計算
        frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
        diff_mean = np.mean(frame_diff)

        # 閾値よりも差分が大きければシーン変更と判断
        if diff_mean > scene_change_threshold:
            is_scene_change = True
            print(
                f"\nシーン変更を検出: フレーム {frame_count}, 差分値: {diff_mean:.2f}"
            )

        prev_frame_gray = frame_gray.copy()

        # トラッカーを更新
        track_success, bbox = face_tracker.update_tracker(frame)

        # 定期的またはトラッキングが失われた場合またはシーン変更時に顔を再検出
        if (
            frame_count % redetection_interval == 0
            or not track_success
            or is_scene_change
        ):
            faces = face_detector.detect_faces(frame, scale_factor)

            if len(faces) > 0:
                best_match_idx = 0

                # カット検出時は顔の特徴量と位置の両方を考慮
                if is_scene_change or not track_success:
                    # 1. 顔の特徴量を使用（ヒストグラム比較）
                    if target_face_hist is not None:
                        best_match_score = -1  # 相関値は-1から1の範囲

                        for i, (fx, fy, fw, fh) in enumerate(faces):
                            # 顔領域を抽出
                            face_roi = frame[fy : fy + fh, fx : fx + fw]
                            if face_roi.size == 0:
                                continue

                            # HSVヒストグラムを計算
                            try:
                                face_roi_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                                face_hist = cv2.calcHist(
                                    [face_roi_hsv],
                                    [0, 1],
                                    None,
                                    [180, 256],
                                    [0, 180, 0, 256],
                                )
                                cv2.normalize(
                                    face_hist, face_hist, 0, 255, cv2.NORM_MINMAX
                                )

                                # ヒストグラム比較（相関法）
                                hist_match = cv2.compareHist(
                                    target_face_hist, face_hist, cv2.HISTCMP_CORREL
                                )

                                if hist_match > best_match_score:
                                    best_match_score = hist_match
                                    best_match_idx = i
                            except:
                                continue

                    # 2. 位置ベースの選択（シーン変更でない場合や特徴量比較が失敗した場合）
                    if is_scene_change == False:
                        last_center_x = x + w // 2
                        last_center_y = y + h // 2
                        min_dist = float("inf")

                        for i, (fx, fy, fw, fh) in enumerate(faces):
                            center_x = fx + fw // 2
                            center_y = fy + fh // 2
                            dist = np.sqrt(
                                (center_x - last_center_x) ** 2
                                + (center_y - last_center_y) ** 2
                            )

                            # 画面サイズに対する相対距離を計算
                            relative_dist = dist / video_processor.frame_width

                            # 距離が一定以下の場合のみ考慮（カットが変わった場合の誤検出防止）
                            if (
                                dist < min_dist and relative_dist < 0.3
                            ):  # 画面幅の30%以内
                                min_dist = dist
                                best_match_idx = i

                    # 3. カットの場合で特徴量比較が不十分な場合は、画面中央や大きさも考慮
                    if is_scene_change:
                        # すでに特徴量で選択済みの場合はスキップ
                        if (
                            target_face_hist is None or best_match_score < 0.5
                        ):  # 相関値が低い場合
                            screen_center_x = video_processor.frame_width / 2
                            screen_center_y = video_processor.frame_height / 2
                            best_score = float("-inf")

                            for i, (fx, fy, fw, fh) in enumerate(faces):
                                # 中央からの距離を計算
                                center_x = fx + fw // 2
                                center_y = fy + fh // 2
                                center_dist = np.sqrt(
                                    (center_x - screen_center_x) ** 2
                                    + (center_y - screen_center_y) ** 2
                                )

                                # 顔の大きさも考慮（大きいほど主要な顔である可能性が高い）
                                face_size = fw * fh

                                # スコアは「顔の大きさ - 中央からの距離」で計算
                                # 中央に近く、大きい顔ほど高スコア
                                score = face_size - center_dist

                                if score > best_score:
                                    best_score = score
                                    best_match_idx = i

                # 選択された顔で更新
                x, y, w, h = faces[best_match_idx]
                bbox = (x, y, w, h)

                # カット検出時や追跡失敗時はトラッカーをリセット
                if is_scene_change or not track_success:
                    face_tracker.reset_tracker(frame, bbox)
                    track_success = True

                    # 新しい対象顔の特徴を更新（適応型追跡）
                    target_face_roi = frame[y : y + h, x : x + w]
                    if target_face_roi.size > 0:
                        try:
                            face_roi_hsv = cv2.cvtColor(
                                target_face_roi, cv2.COLOR_BGR2HSV
                            )
                            new_hist = cv2.calcHist(
                                [face_roi_hsv],
                                [0, 1],
                                None,
                                [180, 256],
                                [0, 180, 0, 256],
                            )
                            cv2.normalize(new_hist, new_hist, 0, 255, cv2.NORM_MINMAX)

                            # 既存のヒストグラムと新しいヒストグラムを徐々に融合
                            if target_face_hist is not None:
                                # 70%古い特徴、30%新しい特徴の割合で更新
                                target_face_hist = (
                                    target_face_hist * 0.7 + new_hist * 0.3
                                )
                            else:
                                target_face_hist = new_hist
                        except:
                            pass

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
        root.withdraw()  # メインウィンドウを非表示に

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
