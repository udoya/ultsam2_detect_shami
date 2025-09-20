"""三味線フレット位置検出と手指先トラッキングの統合システム

SAM2とMediaPipeを使用してフレット位置と指の位置を同時に検出・表示する
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import SAM
from ultralytics.utils.plotting import colors

# 日本語フォントの設定
mpl.rcParams["font.family"] = [
    "DejaVu Sans",
    "Arial",
    "sans-serif",
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


class BoundingBoxSelector:
    """マウスでバウンディングボックスを選択するためのクラス"""

    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.bbox = None

    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベントのコールバック関数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.selecting = True
            self.bbox = None

        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.selecting = False
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                self.bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def draw_selection(self, image):
        """選択中のバウンディングボックスを描画"""
        if self.selecting and self.start_point and self.end_point:
            cv2.rectangle(image, self.start_point, self.end_point, (0, 255, 0), 2)

    def get_bbox(self):
        """現在のバウンディングボックスを取得"""
        return self.bbox


class FPSCounter:
    """FPSカウンタークラス"""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []

    def update(self):
        """現在のタイムスタンプを記録"""
        current_time = time.time()
        self.timestamps.append(current_time)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

    def get_fps(self):
        """現在のFPSを計算"""
        if len(self.timestamps) < 2:
            return 0.0
        time_span = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / max(time_span, 0.001)


class ShamisenFretHandTracker:
    """三味線のフレット位置と手の指先を同時に検出・トラッキングするクラス"""

    def __init__(self, model_name: str = "sam2.1_t.pt", trim_ratio: float = 0.95) -> None:
        """初期化

        Args:
            model_name: 使用するSAMモデル名
            trim_ratio: マスクトリミング用の倍率

        """
        # SAMモデルの初期化
        self.sam_model = SAM(model_name)

        # デバイスの自動選択（Mac対応）
        import torch

        if torch.backends.mps.is_available():
            device = "mps"  # Mac GPU (Metal Performance Shaders)
        elif torch.cuda.is_available():
            device = "cuda:0"  # NVIDIA GPU
        else:
            device = "cpu"  # CPU

        self.sam_model.to(device)
        print(f"SAMモデルを{device}で実行します")

        # トリミング設定
        self.trim_ratio = trim_ratio

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

        # 三味線の音階定義(各弦の開放弦から)
        # 1の糸(細い糸): E4, 2の糸(中糸): B3, 3の糸(太糸): E3
        # 本調子(D-G-D, 1~9, 1が開放弦)
        self.string_notes = {
            "string_3": ["D4", "Eb4", "E4", "F4", "G4", "Ab4", "A4", "Bb4", "C6"],
            "string_2": ["G3", "Ab3", "A3", "Bb3", "C4", "Db4", "D4", "Eb4", "F4", "G4"],
            "string_1": ["D3", "Eb3", "E3", "F3", "G3", "Ab3", "A3", "Bb3", "C4"],
        }

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

            # x座標の最大値の指定倍率を超える部分のマスクを削除
            trimmed_mask_array = self._trim_mask_by_x_coordinate(
                mask_array,
                trim_ratio=self.trim_ratio,
            )

            # マスクの分断部分を接続
            connected_mask = self._connect_mask_segments(trimmed_mask_array)
            connected_mask_array = np.array([connected_mask])

            # マスク画像に追加
            mask_frame[connected_mask > 0] = 255

            # 線分の検出
            line, angle = self._detect_string_line(connected_mask_array)

            if line is not None:
                # 線分をマスクの端まで延長
                extended_line = self._extend_line_to_mask_x_bounds(line, trimmed_mask_array)
                x1, y1, x2, y2 = extended_line
                cv2.line(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                line_info = (extended_line, angle)

            # オーバーレイ画像に半透明でマスクを追加（トリミング後のマスクを使用）
            color = colors(i, bgr=True)
            color_mask = np.zeros_like(frame)
            color_mask[trimmed_mask_array[0] > 0] = color
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

    def estimate_played_notes(
        self,
        fret_positions: list[tuple[float, float, int]],
        hand_result: Any,
        image_shape: tuple[int, int],
    ) -> list[dict]:
        """指の位置とフレット位置から演奏されている音を推定

        Args:
            fret_positions: 計算済みのフレット位置リスト [(x, y, fret_index), ...]
            hand_result: 手の検出結果
            image_shape: 画像の形状 (height, width)

        Returns:
            推定された音のリスト
            [{"finger": str, "fret": int, "string": str, "note": str, "position": tuple}]

        """
        if not fret_positions or hand_result is None:
            print("デバッグ: fret_positions または hand_result が空/None です")
            if not fret_positions:
                print("  - fret_positions が空")
            if hand_result is None:
                print("  - hand_result が None")
            return []

        print("デバッグ: 手の検出結果の確認中...")
        if not hasattr(hand_result, "handedness") or not hasattr(hand_result, "hand_landmarks"):
            print("  - hand_result に handedness または hand_landmarks 属性がありません")
            return []

        print(
            f"  - 検出された手の数: {len(hand_result.handedness) if hand_result.handedness else 0}",
        )

        height, width = image_shape
        print(f"デバッグ: 画像サイズ = {width}x{height}")

        # 指先の位置を取得(左手のみ、人差し指・中指・薬指のみ)
        finger_positions = []
        finger_names = {8: "index", 12: "middle", 16: "ring"}  # 親指(4)と小指(20)を除外

        print("デバッグ: 指先位置の取得開始...")

        for i, (handedness, hand_landmarks) in enumerate(
            zip(hand_result.handedness, hand_result.hand_landmarks),
        ):
            hand_label = handedness[0].display_name
            print(f"  - 手 {i}: {hand_label}")

            if hand_label == "Right":
                print("    右手をスキップ")
                continue

            print("    左手を処理中...")
            for index, landmark in enumerate(hand_landmarks):
                if index in finger_names:
                    visibility = landmark.visibility
                    finger_x = int(landmark.x * width)
                    finger_y = int(landmark.y * height)
                    print(
                        f"      指{index}({finger_names[index]}): 位置=({finger_x},{finger_y}), 可視性={visibility:.3f}",
                    )

                    # 可視性チェックを緩和：位置が画像内にあれば追加
                    if 0 <= finger_x < width and 0 <= finger_y < height:
                        finger_positions.append(
                            {
                                "name": finger_names[index],
                                "position": (finger_x, finger_y),
                                "landmark_index": index,
                                "visibility": visibility,
                            },
                        )
                        print("        -> 追加されました (位置ベース)")
                    else:
                        print(f"        -> 画像外のためスキップ ({finger_x}, {finger_y})")

        print(f"デバッグ: 有効な指先数: {len(finger_positions)}")
        print(fret_positions, "フレット位置の確認...")

        # 指とフレットの対応を判定
        played_notes = []
        tolerance = 100  # ピクセル単位での許容範囲（大きめに設定）

        print(f"デバッグ: フレット位置数: {len(fret_positions)}")
        for i, (fret_x, fret_y, fret_index) in enumerate(fret_positions[:5]):  # 最初の5個だけ表示
            print(f"  フレット{fret_index}: ({fret_x:.1f}, {fret_y:.1f})")

        print(f"デバッグ: 指とフレットの対応判定開始 (許容範囲: {tolerance}px)...")

        for finger in finger_positions:
            finger_x, finger_y = finger["position"]
            closest_fret = None
            min_distance = float("inf")

            print(
                f"  指 {finger['name']} at ({finger_x}, {finger_y}) [可視性: {finger.get('visibility', 0):.3f}]:",
            )

            # 最も近いフレットを見つける
            for fret_x, fret_y, fret_index in fret_positions:
                distance = np.sqrt((finger_x - fret_x) ** 2 + (finger_y - fret_y) ** 2)
                print(f"    フレット{fret_index + 1}への距離: {distance:.1f}px")
                if distance < min_distance:
                    min_distance = distance
                    if distance < tolerance:
                        closest_fret = fret_index + 1

            print(f"    最近フレット: {closest_fret}, 距離: {min_distance:.1f}px")

            # フレットが見つからない場合でも、弦の判定だけ行って開放弦として扱う
            if closest_fret is None and min_distance < tolerance * 2:  # より広い範囲で開放弦を検討
                closest_fret = 1  # 開放弦として扱う
                print("    -> 開放弦として扱います")

            if closest_fret is not None:
                # 弦の判定を行わず、全ての弦の音を候補として出力
                all_possible_notes = []

                # 各弦で該当するフレット位置の音を取得
                for string_name, notes in self.string_notes.items():
                    if closest_fret < len(notes):
                        note = notes[closest_fret]
                        all_possible_notes.append(
                            {
                                "finger": finger["name"],
                                "fret": closest_fret,
                                "string": string_name,
                                "note": note,
                                "position": finger["position"],
                                "distance": min_distance,
                            },
                        )

                # 全ての候補音を追加
                played_notes.extend(all_possible_notes)
                print(f"    -> 音追加 (全弦): {[note['note'] for note in all_possible_notes]}")
            else:
                print("    -> フレットが見つかりません")

        print(f"デバッグ: 最終的な演奏音数: {len(played_notes)}")
        return played_notes

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
        """手のランドマークからバウンディングボックスを計算(左手のみ)"""
        image_width, image_height = image.shape[1], image.shape[0]
        bboxes = []

        for handedness, hand_landmarks in zip(
            detection_result.handedness,
            detection_result.hand_landmarks,
        ):
            # 右手をスキップ(左手のみ処理)
            if handedness[0].display_name == "Right":
                continue

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

        # ランドマークの描画情報（人差し指・中指・薬指のみ強調）
        # landmark_colors = {
        #     4: (128, 128, 128),  # 親指先端（グレー、検出対象外）
        #     8: (128, 0, 255),  # 人差し指先端（紫）
        #     12: (128, 128, 0),  # 中指先端（オリーブ）
        #     16: (192, 192, 192),  # 薬指先端（シルバー）
        #     20: (128, 128, 128),  # 小指先端（グレー、検出対象外）
        # }

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

        # 統一計算メソッドを使用してフレット位置を取得
        fret_positions = self.calculate_fret_positions(line_info)

        # フレット位置計算と描画
        slope_perpendicular = -1 / np.tan(np.radians(angle))

        for tmp_x, tmp_y, i in fret_positions:
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

            # print tmpx, y
            print(f"フレット位置: ({tmp_x:.1f}, {tmp_y:.1f}), フレット番号: {text}")

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

    def draw_estimated_notes(
        self,
        image: np.ndarray,
        played_notes: list[dict],
    ) -> np.ndarray:
        """推定された音を画像上に描画する

        Args:
            image: 描画対象の画像
            played_notes: 推定された音のリスト

        Returns:
            音名が描画された画像

        """
        if not played_notes:
            return image

        # 音名表示用の設定
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # 音名ごとに色を変える
        note_colors = {
            "C": (255, 0, 0),  # 赤
            "C#": (255, 127, 0),  # オレンジ
            "D": (255, 255, 0),  # 黄
            "D#": (127, 255, 0),  # 黄緑
            "E": (0, 255, 0),  # 緑
            "F": (0, 255, 127),  # 青緑
            "F#": (0, 255, 255),  # シアン
            "G": (0, 127, 255),  # 青
            "G#": (0, 0, 255),  # 青
            "A": (127, 0, 255),  # 紫
            "A#": (255, 0, 255),  # マゼンタ
            "B": (255, 0, 127),  # ピンク
        }

        # 指先にシンプルな音名表示
        for note_info in played_notes:
            finger_x, finger_y = note_info["position"]
            note = note_info["note"]

            # 音名の基本名(オクターブ番号を除く)
            note_base = note[:-1] if note[-1].isdigit() else note
            color = note_colors.get(note_base, (255, 255, 255))  # デフォルトは白

            # 指先に小さな円を描画
            cv2.circle(image, (finger_x, finger_y), 3, color, -1)

        # 画像右下にターミナル風リスト表示
        self._draw_notes_terminal_style(image, played_notes, note_colors)

        return image

    def _draw_notes_terminal_style(
        self,
        image: np.ndarray,
        played_notes: list[dict],
        note_colors: dict,
    ) -> None:
        """画像右下にターミナル風の音階リストを描画"""
        if not played_notes:
            return

        height, width = image.shape[:2]

        # ターミナル風背景の設定
        terminal_font = cv2.FONT_HERSHEY_SIMPLEX
        terminal_font_scale = 0.6
        terminal_thickness = 1
        line_height = 25
        padding = 10

        # 指ごとに音をグループ化
        finger_notes = {}
        for note_info in played_notes:
            finger = note_info["finger"]
            if finger not in finger_notes:
                finger_notes[finger] = []
            finger_notes[finger].append(note_info)

        # 表示するテキストを準備
        display_lines = ["=== Detected Notes ==="]
        for finger, notes in finger_notes.items():
            finger_display = {"index": "Index", "middle": "Middle", "ring": "Ring"}
            finger_name = finger_display.get(finger, finger.capitalize())

            # 各弦の音をまとめて表示
            string_notes = {}
            for note in notes:
                string = note["string"]
                if string not in string_notes:
                    string_notes[string] = []
                string_notes[string].append(note["note"])

            display_lines.append(f"{finger_name}: Fret{notes[0]['fret']}")
            for string, notes_list in string_notes.items():
                string_num = string.split("_")[1]
                display_lines.append(f"  S{string_num}: {', '.join(notes_list)}")

        # 背景のサイズを計算
        max_text_width = 0
        for line in display_lines:
            text_size = cv2.getTextSize(
                line,
                terminal_font,
                terminal_font_scale,
                terminal_thickness,
            )[0]
            max_text_width = max(max_text_width, text_size[0])

        terminal_width = max_text_width + padding * 2
        terminal_height = len(display_lines) * line_height + padding * 2

        # 右下の位置を計算
        terminal_x = width - terminal_width - 20
        terminal_y = height - terminal_height - 20

        # ターミナル風背景を描画（半透明の黒）
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (terminal_x, terminal_y),
            (terminal_x + terminal_width, terminal_y + terminal_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

        # 枠線を描画
        cv2.rectangle(
            image,
            (terminal_x, terminal_y),
            (terminal_x + terminal_width, terminal_y + terminal_height),
            (100, 100, 100),
            2,
        )

        # テキストを描画
        for i, line in enumerate(display_lines):
            text_x = terminal_x + padding
            text_y = terminal_y + padding + (i + 1) * line_height

            # ヘッダー行は白、その他は緑色
            if i == 0:
                text_color = (255, 255, 255)  # 白
            elif line.startswith("  S"):
                # 音名部分の色付け
                parts = line.split(": ")
                if len(parts) == 2:
                    # "  S1: " 部分を緑で描画
                    cv2.putText(
                        image,
                        parts[0] + ": ",
                        (text_x, text_y),
                        terminal_font,
                        terminal_font_scale,
                        (0, 255, 0),  # 緑
                        terminal_thickness,
                        cv2.LINE_AA,
                    )

                    # 音名部分を音に応じた色で描画
                    prefix_width = cv2.getTextSize(
                        parts[0] + ": ",
                        terminal_font,
                        terminal_font_scale,
                        terminal_thickness,
                    )[0][0]
                    notes_text = parts[1]

                    # 各音名を個別に色付け
                    note_x_offset = 0
                    for note in notes_text.split(", "):
                        note_base = note[:-1] if note[-1].isdigit() else note
                        note_color = note_colors.get(note_base, (255, 255, 255))

                        cv2.putText(
                            image,
                            note,
                            (text_x + prefix_width + note_x_offset, text_y),
                            terminal_font,
                            terminal_font_scale,
                            note_color,
                            terminal_thickness,
                            cv2.LINE_AA,
                        )

                        note_width = cv2.getTextSize(
                            note,
                            terminal_font,
                            terminal_font_scale,
                            terminal_thickness,
                        )[0][0]
                        note_x_offset += (
                            note_width
                            + cv2.getTextSize(
                                ", ",
                                terminal_font,
                                terminal_font_scale,
                                terminal_thickness,
                            )[0][0]
                        )
                else:
                    cv2.putText(
                        image,
                        line,
                        (text_x, text_y),
                        terminal_font,
                        terminal_font_scale,
                        (0, 255, 0),  # 緑
                        terminal_thickness,
                        cv2.LINE_AA,
                    )
            else:
                text_color = (0, 255, 0)  # 緑
                cv2.putText(
                    image,
                    line,
                    (text_x, text_y),
                    terminal_font,
                    terminal_font_scale,
                    text_color,
                    terminal_thickness,
                    cv2.LINE_AA,
                )

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

        # 演奏音の推定と描画
        if hand_result and line_info:
            # フレット位置を統一計算メソッドで計算
            fret_positions = self.calculate_fret_positions(line_info)

            played_notes = self.estimate_played_notes(
                fret_positions,
                hand_result,
                (frame.shape[0], frame.shape[1]),
            )
            result_image = self.draw_estimated_notes(result_image, played_notes)

            # 推定された音の情報をコンソールに出力
            if played_notes:
                print("推定された演奏音:")
                for note_info in played_notes:
                    print(
                        f"  {note_info['finger']} -> フレット{note_info['fret']} "
                        f"({note_info['string']}) -> {note_info['note']}",
                    )
            else:
                print("演奏音が検出されませんでした。")

        # 結果保存
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/shamisen_fret_hand_{timestamp}.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"結果を保存しました: {output_path}")

        return result_image

    def process_frame(
        self,
        frame: np.ndarray,
        bbox: list[int] | None = None,
        tracking_bbox: list[int] | None = None,
    ) -> tuple[np.ndarray, list[int] | None]:
        """単一フレームを処理してフレット位置と手の検出を行う

        Args:
            frame: 入力フレーム
            bbox: 初期バウンディングボックス (最初のフレームのみ)
            tracking_bbox: 追跡中のバウンディングボックス

        Returns:
            処理結果の画像と新しい追跡ボックス

        """
        # 使用するバウンディングボックスを決定
        current_bbox = bbox if bbox is not None else tracking_bbox

        if current_bbox is None:
            print("警告: バウンディングボックスが指定されていません")
            return frame, None

        # 一時的に画像として保存（SAMの制限のため）
        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        try:
            # 三味線の棹検出
            sam_result, line_info, overlay_frame = self.detect_shamisen_neck(
                temp_image_path,
                current_bbox,
            )

            # 新しい追跡ボックスを取得
            new_tracking_bbox = None
            if sam_result and len(sam_result) > 0 and sam_result[0].boxes is not None:
                if len(sam_result[0].boxes.xyxy) > 0:
                    new_tracking_bbox = sam_result[0].boxes.xyxy[0].cpu().numpy().tolist()

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

            # 演奏音の推定と描画
            if hand_result and line_info:
                # フレット位置を統一計算メソッドで計算
                fret_positions = self.calculate_fret_positions(line_info)

                played_notes = self.estimate_played_notes(
                    fret_positions,
                    hand_result,
                    (frame.shape[0], frame.shape[1]),
                )
                result_image = self.draw_estimated_notes(result_image, played_notes)

            return result_image, new_tracking_bbox

        except Exception as e:
            print(f"フレーム処理中にエラーが発生しました: {e}")
            return frame, tracking_bbox
        finally:
            # 一時ファイルを削除
            if Path(temp_image_path).exists():
                Path(temp_image_path).unlink()

    def process_video(
        self,
        video_path: str,
        initial_bbox: list[int],
        output_path: str | None = None,
        sam_interval: int = 5,
        max_frames: int | None = None,
    ) -> None:
        """動画を処理してフレット位置と手の検出を行う

        Args:
            video_path: 入力動画のパス
            initial_bbox: 初期バウンディングボックス
            output_path: 出力動画のパス
            sam_interval: SAM処理の間隔（フレーム数）
            max_frames: 処理する最大フレーム数（Noneで全フレーム）

        """
        # 動画の読み込み
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
            return

        # 動画の設定取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"動画情報: {width}x{height}, {fps:.2f}fps, {total_frames}フレーム")

        # 出力動画の設定
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("動画処理を開始...")
        frame_count = 0
        tracking_bbox = None
        last_processed_frame = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("すべてのフレームの処理が完了しました")
                    break

                if frame is None:
                    print(f"フレーム {frame_count} をスキップしました（空のフレーム）")
                    continue

                # SAM処理のタイミング判定
                if frame_count % sam_interval == 0:
                    print(f"フレーム {frame_count} でSAM検出を実行中...")

                    # 最初のフレームは初期bboxを使用、以降は追跡bboxを使用
                    bbox_to_use = initial_bbox if frame_count == 0 else None

                    processed_frame, new_tracking_bbox = self.process_frame(
                        frame,
                        bbox_to_use,
                        tracking_bbox,
                    )

                    if new_tracking_bbox is not None:
                        tracking_bbox = new_tracking_bbox
                        print(f"  > 物体を追跡中。新しいbbox: {tracking_bbox}")
                    else:
                        print("  > 警告: 物体を見失いました。追跡を停止します。")
                        tracking_bbox = None

                    last_processed_frame = processed_frame
                else:
                    # SAM処理しないフレームは前回の結果を使用
                    processed_frame = (
                        last_processed_frame if last_processed_frame is not None else frame
                    )

                # 結果の書き込み
                if writer and processed_frame is not None:
                    writer.write(processed_frame)

                frame_count += 1

                # 最大フレーム数の制限
                if max_frames and frame_count >= max_frames:
                    print(f"最大フレーム数 {max_frames} に到達しました")
                    break

                # 進捗表示
                if frame_count % 30 == 0:  # 30フレームごとに進捗表示
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"進捗: {frame_count}/{total_frames} フレーム ({progress:.1f}%)")

        except Exception as e:
            print(f"動画処理中にエラーが発生しました: {e}")
        finally:
            # リソースの解放
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            if output_path:
                print(f"処理完了! 出力動画: {output_path}")
            else:
                print("処理完了!")

    def process_realtime(
        self,
        camera_id: int = 3,
        initial_bbox: list[int] | None = None,
        sam_interval: int = 2,
        output_path: str | None = None,
    ) -> None:
        """リアルタイムカメラqからの映像を処理してフレット位置と手の検出を行う

        Args:
            camera_id: カメラのID（通常は0）
            initial_bbox: 初期バウンディングボックス（Noneの場合は手動設定）
            sam_interval: SAM処理の間隔（フレーム数）
            output_path: 録画保存先パス（Noneで録画しない）

        """
        # カメラの初期化
        print(f"カメラ {camera_id} の初期化を試行中...")
        cap = cv2.VideoCapture(camera_id)

        # カメラが開けるか確認
        if not cap.isOpened():
            print(f"エラー: カメラ {camera_id} を開けませんでした")
            print("利用可能なカメラを確認中...")

            # 他のカメラIDを試す
            for test_id in range(5):
                test_cap = cv2.VideoCapture(test_id)
                if test_cap.isOpened():
                    print(f"カメラ ID {test_id} が利用可能です")
                    test_cap.release()
                else:
                    print(f"カメラ ID {test_id} は利用できません")
            return

        # カメラの詳細情報を表示
        print(f"カメラ {camera_id} を正常に開きました")

        # カメラの設定を試行
        print("カメラの設定を行います...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 実際の設定を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"カメラ情報: {width}x{height}, {fps:.2f}fps")

        # テストフレームの取得
        print("テストフレームを取得中...")
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("エラー: テストフレームの取得に失敗しました")
            cap.release()
            return

        print(f"テストフレーム: {test_frame.shape}, dtype: {test_frame.dtype}")
        print("カメラの初期化が完了しました")
        print("操作:")
        print("  'q' または ESC: 終了")
        print("  's': スクリーンショット保存")
        print("  'r': 録画開始/停止")
        if initial_bbox is None:
            print("  マウスで三味線の棹を囲んでバウンディングボックスを設定してください")

        # 録画用ライター
        writer = None
        is_recording = False

        # バウンディングボックス設定用の変数
        bbox_selector = BoundingBoxSelector()
        current_bbox = initial_bbox
        mouse_callback_set = False

        frame_count = 0
        tracking_bbox = None
        last_processed_frame = None
        fps_counter = FPSCounter()

        # ウィンドウを事前に作成
        cv2.namedWindow("Realtime Shamisen Tracker", cv2.WINDOW_AUTOSIZE)
        print("カメラウィンドウを作成しました")

        try:
            print("リアルタイム処理を開始します...")
            print("カメラフィードを待機中...")

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("フレームの取得に失敗しました")
                    # 少し待ってから再試行
                    cv2.waitKey(10)
                    continue

                # 最初のフレームを取得したら報告
                if frame_count == 0:
                    print(f"最初のフレームを取得しました: {frame.shape}")

                fps_counter.update()
                display_frame = frame.copy()

                # バウンディングボックスが設定されていない場合
                if current_bbox is None:
                    if not mouse_callback_set:
                        print("マウスコールバックを設定中...")
                        cv2.setMouseCallback(
                            "Realtime Shamisen Tracker",
                            bbox_selector.mouse_callback,
                        )
                        mouse_callback_set = True
                        print("マウスコールバックを設定完了")

                    # バウンディングボックス選択中の描画
                    bbox_selector.draw_selection(display_frame)
                    current_bbox = bbox_selector.get_bbox()

                    if current_bbox is not None:
                        print(f"バウンディングボックスが設定されました: {current_bbox}")

                    # 指示文を表示
                    cv2.putText(
                        display_frame,
                        "Select shamisen neck with mouse",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        "Drag to select area, then release mouse",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # バウンディングボックスが設定済み - 処理開始
                    if frame_count % sam_interval == 0:
                        print(f"フレーム {frame_count} でSAM検出を実行中...")

                        # 最初のSAM実行時は設定されたbboxを使用
                        bbox_to_use = current_bbox if tracking_bbox is None else None

                        processed_frame, new_tracking_bbox = self.process_frame(
                            frame,
                            bbox_to_use,
                            tracking_bbox,
                        )

                        if new_tracking_bbox is not None:
                            tracking_bbox = new_tracking_bbox
                            print("  > 物体を追跡中")
                        else:
                            print("  > 警告: 物体を見失いました")

                        last_processed_frame = processed_frame
                    else:
                        # SAM処理しないフレームは前回の結果を使用
                        processed_frame = (
                            last_processed_frame if last_processed_frame is not None else frame
                        )

                    display_frame = processed_frame

                # FPS表示
                current_fps = fps_counter.get_fps()
                cv2.putText(
                    display_frame,
                    f"FPS: {current_fps:.1f}",
                    (display_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # 録画状態表示
                if is_recording:
                    cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(
                        display_frame,
                        "REC",
                        (50, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # 画面表示を確実に行う
                cv2.imshow("Realtime Shamisen Tracker", display_frame)

                # 画面の更新を強制
                cv2.waitKey(1)

                # 録画処理
                if is_recording and writer is not None:
                    writer.write(display_frame)

                # キー入力処理
                key = cv2.waitKey(30) & 0xFF  # 30ms待機でキー入力検出
                if key == ord("q") or key == 27:  # 'q' または ESC
                    break
                if key == ord("s"):  # スクリーンショット
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"スクリーンショットを保存: {screenshot_path}")
                elif key == ord("r"):  # 録画切り替え
                    if not is_recording:
                        # 録画開始
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        record_path = output_path or f"recording_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(record_path, fourcc, fps, (width, height))
                        is_recording = True
                        print(f"録画開始: {record_path}")
                    else:
                        # 録画停止
                        if writer:
                            writer.release()
                            writer = None
                        is_recording = False
                        print("録画停止")

                frame_count += 1

        except Exception as e:
            print(f"リアルタイム処理中にエラーが発生しました: {e}")
        finally:
            # リソースの解放
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("リアルタイム処理を終了しました")

    def _trim_mask_by_x_coordinate(
        self,
        mask_array: np.ndarray,
        trim_ratio: float = 0.95,
    ) -> np.ndarray:
        """x座標の最大値の指定倍率を超える部分のマスクを削除

        Args:
            mask_array: 入力マスク配列
            trim_ratio: 最大x座標にかける倍率(デフォルト: 0.95)

        Returns:
            トリミングされたマスク配列

        """
        # 定数定義
        MASK_3D_CHANNELS = 3
        MASK_2D_CHANNELS = 2

        # マスクの次元を確認
        if mask_array.ndim == MASK_3D_CHANNELS and mask_array.shape[0] == 1:
            mask_2d = mask_array[0]
            return_3d = True
        elif mask_array.ndim == MASK_2D_CHANNELS:
            mask_2d = mask_array
            return_3d = False
        else:
            mask_2d = np.squeeze(mask_array)
            return_3d = mask_array.ndim == MASK_3D_CHANNELS

        # マスクが存在するピクセルの座標を取得
        y_indices, x_indices = np.where(mask_2d > 0)

        if len(x_indices) == 0:
            # マスクが空の場合はそのまま返す
            print("デバッグ: マスクが空のため、トリミングをスキップします")
            return mask_array

        # x座標の最大値と閾値を計算
        max_x = np.max(x_indices)
        min_x = np.min(x_indices)

        x_threshold = ((max_x - min_x) * trim_ratio) + min_x

        # print(f"デバッグ: マスクx座標範囲: {np.min(x_indices)} - {max_x}")
        # print(f"デバッグ: x座標閾値: {x_threshold:.1f} (最大値の{trim_ratio}倍)")

        # 新しいマスクを作成
        trimmed_mask = mask_2d.copy()

        # 閾値を超えるx座標の部分をゼロにする
        trimmed_mask[:, int(x_threshold) :] = 0

        # トリミング後の統計
        trimmed_y_indices, trimmed_x_indices = np.where(trimmed_mask > 0)
        if len(trimmed_x_indices) > 0:
            print(
                f"デバッグ: トリミング後x座標範囲: "
                f"{np.min(trimmed_x_indices)} - {np.max(trimmed_x_indices)}",
            )
            removed_pixels = len(x_indices) - len(trimmed_x_indices)
            removal_percentage = removed_pixels / len(x_indices) * 100
            print(
                f"デバッグ: 削除されたピクセル数: {removed_pixels} ({removal_percentage:.1f}%)",
            )
        else:
            print("警告: トリミング後にマスクが空になりました")

        # 元の次元に合わせて返す
        if return_3d:
            return np.array([trimmed_mask])
        return trimmed_mask

    def calculate_fret_positions(
        self,
        line_info: tuple[list[int], float],
    ) -> list[tuple[float, float, int]]:
        """フレット位置を計算する統一メソッド

        Args:
            line_info: 線分情報 (line, angle)

        Returns:
            フレット位置のリスト [(x, y, fret_index), ...]

        """
        if line_info is None:
            return []

        line, angle = line_info
        x1, y1, x2, y2 = line

        # 弦の長さ計算(画像上での三味線の棹の長さを推定)
        x_distance_mask = abs(x2 - x1)
        x_length_string = (x_distance_mask * 4) / 3

        # ブリッジ位置計算
        x_bridge = x2 - x_length_string
        y_bridge = y1 + (x_bridge - x1) * np.tan(np.radians(angle))

        # フレット位置を計算
        fret_positions = []
        for i in range(len(self.fret_ratios) - 1):
            tmp_x_distance = x_length_string * self.fret_ratios[i]
            tmp_x = x_bridge + tmp_x_distance
            tmp_y = y_bridge + (tmp_x - x_bridge) * np.tan(np.radians(angle))
            fret_positions.append((tmp_x, tmp_y, i))

        return fret_positions


def main() -> None:
    """メイン処理 - リアルタイムカメラ処理モード"""
    # パラメータ設定
    trim_ratio = 0.9  # マスクトリミング倍率 (0.0-1.0)
    sam_interval = 10  # SAM処理の間隔 (フレーム数)

    # リアルタイム処理用設定
    camera_id = 3  # カメラID (通常は0)
    initial_bbox = None  # 手動でバウンディングボックスを設定
    output_path = None  # 録画しない場合はNone

    # トラッカーの初期化
    tracker = ShamisenFretHandTracker(
        model_name="sam2.1_t.pt",
        trim_ratio=trim_ratio,
    )

    print("三味線フレット・手指先リアルタイム検出システム")
    print("=" * 50)
    print(f"マスクトリミング倍率: {trim_ratio}")
    print(f"SAM処理間隔: {sam_interval} フレーム")
    print(f"カメラID: {camera_id}")
    print()
    print("使用方法:")
    print("1. カメラ画面で三味線の棹をマウスで囲んでください")
    print("2. 'q' または ESC キーで終了")
    print("3. 's' キーでスクリーンショット保存")
    print("4. 'r' キーで録画開始/停止")

    # リアルタイム処理の実行
    try:
        tracker.process_realtime(
            camera_id=camera_id,
            initial_bbox=initial_bbox,
            sam_interval=sam_interval,
            output_path=output_path,
        )
    except KeyboardInterrupt:
        print("\nキーボード割り込みで終了しました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


def main_video() -> None:
    """動画ファイル処理モード (従来の処理)"""
    # 設定パラメータ
    video_path = "data/seg.mp4"  # 入力動画のパス
    initial_bbox = [600, 300, 1900, 800]  # 初期バウンディングボックス
    output_path = "data/output_shamisen_v3a.mp4"  # 出力動画のパス
    sam_interval = 10
    max_frames = 180
    trim_ratio = 0.86  # マスクトリミング倍率: 0.0-1.0

    tracker = ShamisenFretHandTracker(model_name="sam2.1_t.pt", trim_ratio=trim_ratio)

    print(f"設定: trim_ratio = {trim_ratio}")
    print("三味線のフレット位置と手の検出(動画)を開始...")
    tracker.process_video(
        video_path=video_path,
        initial_bbox=initial_bbox,
        output_path=output_path,
        sam_interval=sam_interval,
        max_frames=max_frames,
    )

    print("動画処理が完了しました。")


def main_image() -> None:
    """画像処理モード(従来の処理)"""
    # 設定パラメータ
    trim_ratio = 0.9  # マスクトリミング倍率（0.0-1.0）

    # 初期化
    tracker = ShamisenFretHandTracker(model_name="sam2.1_t.pt", trim_ratio=trim_ratio)

    # 設定
    image_path = "./data/image.png"
    bbox = [300, 20, 780, 380]
    # 出力ディレクトリ設定
    timestamp_mmdd = time.strftime("%m%d")
    output_dir = f"./out/{timestamp_mmdd}/shamisen_fret_hand"

    print(f"設定: trim_ratio = {trim_ratio}")
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
    # リアルタイムカメラ処理をデフォルトに変更
    # 動画ファイル処理を実行したい場合は main_video() を呼び出してください
    main()
