# video_path = "data/forsegment.mp4"  # <<< 入力動画のパス
# initial_bbox = [600, 300, 1200, 500]


video_path = "data/ForMovie3.mp4"
initial_bbox = [600, 400, 1700 - 600, 800 - 400]

import cv2
import numpy as np


def visualize_bbox_on_first_frame():
    """動画の1フレーム目にinitial_bboxを描画して表示する"""
    # 動画を開く
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: 動画ファイルを開けませんでした: {video_path}")
        return

    # 1フレーム目を読み込み
    ret, frame = cap.read()
    if not ret:
        print("Error: 1フレーム目を読み込めませんでした")
        cap.release()
        return

    # bboxを描画 (x, y, w, h)
    x, y, w, h = initial_bbox

    # 矩形を描画（緑色、線の太さ3）
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # bboxの座標情報をテキストで描画
    text = f"bbox: [{x}, {y}, {w}, {h}]"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print(f"動画サイズ: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Initial bbox: x={x}, y={y}, w={w}, h={h}")

    # 画像を表示
    cv2.imshow("First Frame with Initial BBox", frame)

    # キー入力を待つ（何かキーを押すと終了）
    print("何かキーを押すと終了します...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()


if __name__ == "__main__":
    visualize_bbox_on_first_frame()
