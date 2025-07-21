import cv2
import torch
from ultralytics import SAM


def main():
    """SAM2モデルを使用して動画内の指定したオブジェクトを追跡するメイン関数"""
    # --- 1. 設定 ---
    model = SAM("sam2_t.pt")

    # 入力と出力のビデオパス
    video_path = "data/ForMovie3.mp4"
    initial_bbox = [600, 400, 1500, 800]

    # video_path = "data/forsegment.mp4"  # <<< 入力動画のパス
    # bbox = [300, 20, 780, 380]

    output_path = "data/output_test.mp4"  # <<< 出力先のパス

    # --- 2. 動画のセットアップ ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 動画の情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 動画書き出し用の設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing started... Press 'q' to quit.")
    frame_count = 0

    # --- 3. フレームごとの処理ループ ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ループの最初の回だけ、BBoxを指定して推論
        if frame_count == 0:
            print("Processing the first frame with the initial bounding box...")
            results = model.predict(frame, bboxes=initial_bbox, stream=True)
        else:
            # 2フレーム目以降はBBoxの指定は不要
            # stream=Trueの効果で、モデルが文脈を維持して追跡する
            results = model.predict(frame, stream=True)

        # 推論結果:セグメンテーションマスク をフレームに描画
        annotated_frame = results[0].plot()

        # 処理したフレームを動画ファイルに書き込む
        writer.write(annotated_frame)

        # リアルタイムで処理結果を表示: 任意
        cv2.imshow("SAM2 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1
        print(f"Processed frame {frame_count}...")

    # --- 4. 終了処理 ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"✨ Processing finished! Output video saved to {output_path}")


if __name__ == "__main__":
    main()
