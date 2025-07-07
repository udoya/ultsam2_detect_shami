#!/usr/bin/env python3
"""音推定機能のテスト用スクリプト"""

from shamisen_fret_hand_tracker_v2 import ShamisenFretHandTracker


def main():
    """テスト実行"""
    print("=== 音推定機能テスト開始 ===")

    # トラッカー初期化
    tracker = ShamisenFretHandTracker(model_name="sam2.1_t.pt")

    # 設定
    image_path = "./data/first_frame.jpg"
    bbox = [600, 400, 1500, 800]  # 三味線の棹のバウンディングボックス

    # 処理実行
    print("処理開始...")
    result_image = tracker.process_image(image_path, bbox, None)  # 保存はしない

    print("=== テスト完了 ===")


if __name__ == "__main__":
    main()
