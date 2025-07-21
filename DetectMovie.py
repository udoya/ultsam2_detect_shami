import cv2
import numpy as np  # OpenCVの描画用にインポート
import torch
from ultralytics import SAM


def main():
    # --- 1. 基本設定 ---
    try:
        # モデルのロードとデバイスの指定
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = SAM("sam2.1_b.pt").to(device)

        # 入出力ビデオパス
        # video_path = "data/output.mp4"
        # initial_bbox = [600, 400, 1500, 800]
        tracking_bbox = None  # 追跡中のBBoxをここに入れる

        video_path = "data/forsegment.mp4"  # <<< 入力動画のパス
        initial_bbox = [600, 300, 1900, 800]

        output_path = "data/output2.mp4"  # <<< 出力先のパス

        # 最初のフレームで追跡したい物体のBBox (x1, y1, x2, y2)

    except Exception as e:
        print(f"Error during setup: {e}")
        return

    # --- 2. 動画のセットアップ ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing started... Press 'q' to quit.")
    frame_count = 0
    sam_interval = 2
    last_annotated_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing all frames or failed to read frame.")
            break

        if frame is None:
            print(f"Skipping empty frame at position {frame_count}.")
            continue
        annotated_frame = None

        if frame_count % sam_interval == 0:
            print(f"--- Running detection on frame {frame_count} ---")

            prompt_bbox = initial_bbox if frame_count == 0 else tracking_bbox

            try:
                # 最初のフレーム(0)ではBBoxを使い、それ以降の検知フレームでは自動追跡
                # results_generator = model.predict(frame, bboxes=prompt_bbox, stream=True)
                results_generator = model.predict(
                    frame,
                    bboxes=[prompt_bbox],
                    stream=True,
                )

                # ジェネレータから結果を取り出して描画
                for result in results_generator:
                    annotated_frame = result.plot()
                    last_annotated_frame = annotated_frame

                    # 結果から新しいBBoxを取得して、次のプロンプト用に更新する
                    if result.boxes and len(result.boxes.xyxy) > 0:
                        # 複数の物体が検出された場合も、最初のものを使う
                        new_bbox = result.boxes.xyxy[0].cpu().numpy().tolist()
                        tracking_bbox = new_bbox  # ここで追跡BBoxを更新
                        found_object = True
                        print(f"  > Object tracked. New bbox: {tracking_bbox}")
                    else:
                        # もしBBoxが取得できなかったら、追跡失敗
                        tracking_bbox = None

                    if not found_object:
                        print("  > Warning: Object lost. Tracking stopped.")
                        tracking_bbox = None  # 追跡失敗

            except Exception as e:
                print(f"An error occurred during prediction on frame {frame_count}: {e}")
                tracking_bbox = None  # エラー時も追跡をリセット
        else:
            annotated_frame = frame.copy() if last_annotated_frame is not None else frame

        # 描画されたフレームがあれば書き込み、なければ元のフレームを書き込む
        if annotated_frame is not None and annotated_frame.size > 0:
            writer.write(annotated_frame)
        else:
            # 最初の数フレームなど、まだ何も描画されていない場合は元のフレームを書き込む
            writer.write(frame)

        frame_count += 1
        if frame_count > 60:
            break

    # --- 4. 終了処理 ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"✨ Processing finished! Output video saved to {output_path}")


if __name__ == "__main__":
    main()
