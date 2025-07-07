"""三味線フレット位置検出と手指先トラッキングの統合システム

SAM2とMediaPipeを使用してフレット位置と指の位置を同時に検出・表示する
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from ultralytics import SAM
from ultralytics.utils.plotting import colors

# 日本語フォントの設定
matplotlib.rcParams["font.family"] = [
    "DejaVu Sans",
    "Arial Unicode MS",
    "Yu Gothic",
    "Meiryo",
    "Takao",
    "IPAexGothic",
    "IPAPGothic",
    "VL PGothic",
    "Noto Sans CJK JP",
]

# MediaPipeの条件付きインポート
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("警告: MediaPipeが利用できません。手の検出機能は無効になります。")

import MyUtils


class ShamisenFretHandTracker:
    """三味線のフレット位置と手の指先を同時に検出・トラッキングするクラス"""

    def __init__(self, model_name: str = "sam2.1_t.pt") -> None:
        """初期化

        Args:
            model_name: 使用するSAMモデル名

        """
        # SAMモデルの初期化
        self.sam_model = SAM(model_name)
        self.sam_model.to("cuda:0")  # GPUに移動

        # MediaPipe Hand Landmarkerの初期化
        self.hand_detector = None
        if MEDIAPIPE_AVAILABLE:
            self._init_hand_detector()

        # フレット位置計算用の比率(平均律)
        self.fret_ratios = [
            1,
            0.9583333333,
            0.9166666667,
            0.875,
            0.7916666667,
            0.75,
            0.7083333333,
            0.6666666667,
            0.5833333333,
            0.5,
            0.4791666667,
            0.4583333333,
            0.4375,
            0.3958333333,
            0.375,
            0.3541666667,
            0.3333333333,
            0.2916666667,
            0.25,
            0,
        ]

        # フレット位置のラベル
        self.fret_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def _init_hand_detector(self) -> None:
        """MediaPipe Hand Landmarkerの初期化"""
        model_path = Path("./model/hand_landmarker_float16.task")

        if not model_path.exists():
            print(f"警告: {model_path} が見つかりません。手の検出機能は無効になります。")
            return

        # HandLandmarker設定
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,  # 最大2手まで検出
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

    def detect_shamisen_neck(self, image_path: str, bbox: list[int]) -> tuple[Any, Any, np.ndarray]:
        """三味線の棹をセグメンテーションして線分を検出

        Args:
            image_path: 画像ファイルのパス
            bbox: バウンディングボックス [x1, y1, x2, y2]

        Returns:
            セグメンテーション結果、線分情報、オーバーレイ画像

        """
        # SAMでセグメンテーション実行
        result = self.sam_model.predict(
            source=image_path,
            bboxes=[bbox],
            save=False,
            verbose=False,
        )

        # 線分検出処理
        overlay_frame, mask_frame, line_info = self._process_mask_with_line_detection(
            image_path,
            result,
        )

        return result, line_info, overlay_frame

    def _process_mask_with_line_detection(
        self,
        image_path: str,
        result: Any,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """マスクから線分検出を行う"""
        # 元画像の読み込み
        frame = cv2.imread(image_path)
        frame_height, frame_width = frame.shape[:2]

        # マスク情報を取得
        masks = result[0].masks
        if masks is None:
            print("マスクが検出されませんでした")
            return frame, np.zeros_like(frame), None

        # マスク画像とオーバーレイ画像の作成
        mask_frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
        overlay_frame = frame.copy()
        line_info = None

        # 各マスクを処理
        for i, mask in enumerate(masks):
            # マスクをNumPy配列に変換
            mask_array = mask.data.cpu().numpy()

            # マスクの分断部分を接続
            connected_mask = self._connect_mask_segments(mask_array)
            connected_mask_array = np.array([connected_mask])

            # マスク画像に追加
            mask_frame[connected_mask > 0] = 255

            # 線分の検出
            line, angle = self._detect_string_line(connected_mask_array)

            if line is not None:
                # 線分をマスクの端まで延長
                extended_line = self._extend_line_to_mask_x_bounds(line, mask_array)
                x1, y1, x2, y2 = extended_line
                cv2.line(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                line_info = (extended_line, angle)

            # オーバーレイ画像に半透明でマスクを追加
            color = colors(i, bgr=True)
            color_mask = np.zeros_like(frame)
            color_mask[mask_array[0] > 0] = color
            overlay_frame = cv2.addWeighted(overlay_frame, 1.0, color_mask, 0.5, 0)

        mask_frame_color = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
        return overlay_frame, mask_frame_color, line_info

    def _connect_mask_segments(self, mask_array_input: np.ndarray) -> np.ndarray:
        """マスクの分断部分を接続する"""
        # 定数定義
        MASK_3D_CHANNELS = 3
        MASK_2D_CHANNELS = 2

        if mask_array_input.ndim == MASK_3D_CHANNELS and mask_array_input.shape[0] == 1:
            processed_mask = mask_array_input[0]
        elif mask_array_input.ndim == MASK_2D_CHANNELS:
            processed_mask = mask_array_input
        else:
            processed_mask = np.squeeze(mask_array_input)
            if processed_mask.ndim != MASK_2D_CHANNELS:
                error_msg = f"Unsupported mask shape: {processed_mask.shape}"
                raise ValueError(error_msg)

        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(processed_mask.astype(np.uint8), kernel, iterations=1)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
        return eroded_mask

    def _weighted_pca(self, mask_array: np.ndarray) -> tuple[Any, Any, Any]:
        """重み付きPCAを使用してマスクの主成分を計算"""
        y_indices, x_indices = np.where(mask_array[0] > 0)
        points = np.column_stack((x_indices, y_indices))

        if len(points) == 0:
            return None, None, None

        mean = np.mean(points, axis=0)
        distances = np.linalg.norm(points - mean, axis=1)
        max_dist = np.max(distances)
        weights = np.ones(len(points)) if max_dist == 0 else 1.0 - (distances / max_dist)

        weighted_mean = np.average(points, axis=0, weights=weights)

        cov = np.zeros((2, 2))
        for pt, w in zip(points, weights):
            diff = (pt - weighted_mean).reshape(-1, 1)
            cov += w * (diff @ diff.T)
        cov /= weights.sum()

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        main_axis_idx = np.argmax(eigenvalues)
        main_eigenvector = eigenvectors[:, main_axis_idx]

        return weighted_mean, main_eigenvector, eigenvalues[main_axis_idx]

    def _detect_string_line(self, mask_array: np.ndarray) -> tuple[list[int] | None, float | None]:
        """マスクから線分を検出"""
        mean, eigenvector, eigenvalue = self._weighted_pca(mask_array)

        if mean is not None:
            line_length = np.sqrt(eigenvalue * 4)
            pt1 = mean + eigenvector * line_length
            pt2 = mean - eigenvector * line_length
            line_angle = np.degrees(np.arctan2(eigenvector[1], eigenvector[0]))
            return [int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1])], line_angle

        return None, None

    def _extend_line_to_mask_x_bounds(self, line: list[int], mask_array: np.ndarray) -> list[int]:
        """マスクのx座標の範囲に基づいて線分を延長"""
        x1, y1, x2, y2 = line

        y_indices, x_indices = np.where(mask_array[0] > 0)
        if len(x_indices) == 0:
            return line

        min_x = np.min(x_indices)
        max_x = np.max(x_indices)

        if x2 - x1 == 0:
            slope = float("inf")
            y_at_min_x = y1
            y_at_max_x = y1
        else:
            slope = (y2 - y1) / (x2 - x1)
            y_at_min_x = y1 + slope * (min_x - x1)
            y_at_max_x = y1 + slope * (max_x - x1)

        # 定数定義
        MASK_2D_DIMENSIONS = 2

        height, width = (
            mask_array[0].shape
            if len(mask_array[0].shape) == MASK_2D_DIMENSIONS
            else mask_array[0].shape[:MASK_2D_DIMENSIONS]
        )
        y_at_min_x = max(0, min(y_at_min_x, height - 1))
        y_at_max_x = max(0, min(y_at_max_x, height - 1))

        return [int(min_x), int(y_at_min_x), int(max_x), int(y_at_max_x)]

    def detect_hands(self, frame: np.ndarray) -> tuple[Any, list[list[int]]]:
        """手の検出を行う

        Args:
            frame: 入力画像

        Returns:
            検出結果とバウンディングボックスのリスト

        """
        if not MEDIAPIPE_AVAILABLE or self.hand_detector is None:
            return None, []

        # MediaPipe用に画像を変換
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        )

        # 手の検出実行
        detection_result = self.hand_detector.detect(rgb_frame)

        # バウンディングボックス計算
        bboxes = self._calc_bounding_rect(frame, detection_result)

        return detection_result, bboxes

    def _calc_bounding_rect(self, image: np.ndarray, detection_result: Any) -> list[list[int]]:
        """手のランドマークからバウンディングボックスを計算（左手のみ）"""
        image_width, image_height = image.shape[1], image.shape[0]
        bboxes = []

        for handedness, hand_landmarks in zip(
            detection_result.handedness,
            detection_result.hand_landmarks,
        ):
            # 右手をスキップ（左手のみ処理）

            landmark_array = np.empty((0, 2), int)
            for landmark in hand_landmarks:
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                landmark_point = np.array((landmark_x, landmark_y))
                landmark_array = np.append(landmark_array, [landmark_point], axis=0)

            x, y, w, h = cv2.boundingRect(landmark_array)
            bboxes.append([x, y, x + w, y + h])

        return bboxes

    def draw_hands(
        self,
        image: np.ndarray,
        detection_result: Any,
        bboxes: list[list[int]],
    ) -> np.ndarray:
        """手のランドマークとバウンディングボックスを描画"""
        if detection_result is None:
            return image

        image_width, image_height = image.shape[1], image.shape[0]

        # ランドマークの描画情報
        landmark_colors = {
            4: (0, 255, 255),  # 親指先端
            8: (128, 0, 255),  # 人差し指先端
            12: (128, 128, 0),  # 中指先端
            16: (192, 192, 192),  # 薬指先端
            20: (220, 20, 60),  # 小指先端
        }

        # 接続線の情報
        line_info_list = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],  # 親指
            [0, 5],
            [5, 6],
            [6, 7],
            [7, 8],  # 人差し指
            [0, 9],
            [9, 10],
            [10, 11],
            [11, 12],  # 中指
            [0, 13],
            [13, 14],
            [14, 15],
            [15, 16],  # 薬指
            [0, 17],
            [17, 18],
            [18, 19],
            [19, 20],  # 小指
        ]

        for handedness, hand_landmarks, bbox in zip(
            detection_result.handedness,
            detection_result.hand_landmarks,
            bboxes,
        ):
            # 右手をスキップ（左手のみ表示）
            if handedness[0].display_name == "Right":
                continue

            # ランドマーク座標を整理
            landmark_dict = {}
            for index, landmark in enumerate(hand_landmarks):
                if landmark.visibility < 0 or landmark.presence < 0:
                    continue
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                landmark_dict[index] = [landmark_x, landmark_y]

            # 接続線描画
            for line_info in line_info_list:
                if line_info[0] in landmark_dict and line_info[1] in landmark_dict:
                    cv2.line(
                        image,
                        tuple(landmark_dict[line_info[0]]),
                        tuple(landmark_dict[line_info[1]]),
                        (220, 220, 220),
                        2,
                        cv2.LINE_AA,
                    )

            # 指先のランドマーク強調表示
            for index, landmark in landmark_dict.items():
                color = landmark_colors.get(index, (0, 255, 0))
                radius = 8 if index in landmark_colors else 3
                cv2.circle(
                    image,
                    (landmark[0], landmark[1]),
                    radius,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

            # バウンディングボックス描画
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 左右表示
            cv2.putText(
                image,
                handedness[0].display_name,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return image

    def draw_fret_positions(
        self,
        image: np.ndarray,
        line_info: tuple[list[int], float],
    ) -> np.ndarray:
        """フレット位置を描画"""
        if line_info is None:
            return image

        line, angle = line_info
        x1, y1, x2, y2 = line

        # 弦の長さ計算(画像上での三味線の棹の長さを推定)
        x_distance_mask = abs(x2 - x1)
        x_length_string = (x_distance_mask * 4) / 3

        # ブリッジ位置計算
        x_bridge = x2 - x_length_string
        y_bridge = y1 + (x_bridge - x1) * np.tan(np.radians(angle))

        # ブリッジ位置描画
        cv2.circle(image, (int(x_bridge), int(y_bridge)), 8, (0, 255, 0), -1)
        cv2.putText(
            image,
            "Bridge",
            (int(x_bridge) + 10, int(y_bridge) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # フレット位置計算と描画
        slope_perpendicular = -1 / np.tan(np.radians(angle))

        for i in range(len(self.fret_ratios) - 1):
            tmp_x_distance = x_length_string * self.fret_ratios[i]
            tmp_x = x_bridge + tmp_x_distance
            tmp_y = y_bridge + (tmp_x - x_bridge) * np.tan(np.radians(angle))

            # フレット番号のラベル
            text_quo = i // len(self.fret_labels)
            text_mod = i % len(self.fret_labels)
            text_dot = "*" * text_quo
            text = self.fret_labels[text_mod] + text_dot

            # フレット位置描画
            cv2.circle(image, (int(tmp_x), int(tmp_y)), 6, (0, 0, 255), -1)
            cv2.putText(
                image,
                text,
                (int(tmp_x) - 10, int(tmp_y) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            # フレットに垂直な線を描画
            b = tmp_y - tmp_x * slope_perpendicular
            x_start = tmp_x - 25
            x_end = tmp_x + 25
            y_start = slope_perpendicular * x_start + b
            y_end = slope_perpendicular * x_end + b
            cv2.line(
                image,
                (int(x_start), int(y_start)),
                (int(x_end), int(y_end)),
                (255, 0, 255),
                2,
            )

        return image

    def process_image(
        self,
        image_path: str,
        bbox: list[int],
        output_dir: str | None = None,
    ) -> np.ndarray:
        """画像を処理してフレット位置と手の検出を行う

        Args:
            image_path: 画像ファイルのパス
            bbox: 三味線の棹のバウンディングボックス
            output_dir: 結果の保存先ディレクトリ

        Returns:
            処理結果の画像

        """
        # 三味線の棹検出
        sam_result, line_info, overlay_frame = self.detect_shamisen_neck(image_path, bbox)

        # 元画像読み込み
        frame = cv2.imread(image_path)

        # 手の検出
        hand_result, hand_bboxes = self.detect_hands(frame)

        # 結果画像の作成
        result_image = overlay_frame.copy()

        # フレット位置描画
        if line_info:
            result_image = self.draw_fret_positions(result_image, line_info)

        # 手のランドマーク描画
        if hand_result:
            result_image = self.draw_hands(result_image, hand_result, hand_bboxes)

        # 結果保存
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/shamisen_fret_hand_{timestamp}.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"結果を保存しました: {output_path}")

        return result_image


def main() -> None:
    """メイン処理"""
    # 初期化
    tracker = ShamisenFretHandTracker(model_name="sam2.1_t.pt")

    # 設定
    image_path = "./data/first_frame.jpg"
    bbox = [600, 400, 1500, 800]  # 三味線の棹のバウンディングボックス

    # 出力ディレクトリ設定
    timestamp_mmdd = time.strftime("%m%d")
    output_dir = f"./out/{timestamp_mmdd}/shamisen_fret_hand"

    # 処理実行
    print("三味線のフレット位置と手の検出を開始...")
    result_image = tracker.process_image(image_path, bbox, output_dir)

    # 結果表示
    plt.figure(figsize=(20, 12))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Shamisen Fret Position Detection + Left Hand Tracking", fontsize=16)
    plt.axis("on")
    plt.show()

    print("処理が完了しました。")


if __name__ == "__main__":
    main()
