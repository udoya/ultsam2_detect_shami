# import
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image as PILImage
# from ultralytics import SAM
# from ultralytics.utils.plotting import colors

# from IPython.display import Image, display


def get_first_frame(
    video_path: str,
    output_path: str = "first_frame.jpg",
) -> str | None:
    """動画の最初のフレームを抽出して保存する

    Args:
        video_path (str): 動画ファイルのパス
        output_path (str): 保存するフレームのパス
    Returns:
        str: 保存したフレームのパス

    """
    cap = cv2.VideoCapture()
    cap.open(str(video_path))  # Open the video file
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(output_path, frame)
        print(f"最初のフレームを {output_path} に保存しました")
        return output_path
    print("フレームの抽出に失敗しました")
    return None


def my_make_dir(dir_name: str) -> None:
    """ディレクトリを作成する関数

    Returns:
        None

    """
    # 東京のタイムゾーン情報を取得
    tokyo_tz = ZoneInfo("Asia/Tokyo")

    timestamp_ymd = datetime.now(tz=tokyo_tz).strftime("%Y%m%d")
    if not Path("out").exists():
        Path("out").mkdir(parents=True)
    if not Path("out", timestamp_ymd).exists():
        Path("out", timestamp_ymd).mkdir(parents=True)
    if not Path("out", timestamp_ymd, dir_name).exists():
        Path("out", timestamp_ymd, dir_name).mkdir(parents=True)
