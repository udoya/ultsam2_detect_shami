#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import argparse
from typing import List, Any, Dict, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp  # type:ignore
from mediapipe.tasks import python  # type:ignore
from mediapipe.tasks.python import vision  # type:ignore

from utils import CvFpsCalc
from utils.download_file import download_file
import matplotlib.pyplot as plt

import torch
from segment_anything import sam_model_registry, SamPredictor
import datetime
import os



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)
    parser.add_argument("--unuse_mirror", action="store_true")
    parser.add_argument(
        "--model",
        type=int,
        choices=[0],
        default=0,
        help="""
        0:HandLandmarker (full)
        """,
    )
    parser.add_argument(
        "--num_hands",
        type=int,
        default=2,
    )
    parser.add_argument("--use_world_landmark", action="store_true")

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args = get_args()

    cap_device: Union[int, str] = args.device
    cap_width: int = args.width
    cap_height: int = args.height
    unuse_mirror: bool = args.unuse_mirror
    use_world_landmark: bool = args.use_world_landmark
    model: int = args.model
    num_hands: int = args.num_hands

    if args.video is not None:
        cap_device = args.video

    model_url: List[str] = [
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    ]

    # ダウンロードファイル名生成
    model_name: str = model_url[model].split("/")[-1]
    quantize_type: str = model_url[model].split("/")[-3]
    split_name: List[str] = model_name.split(".")
    model_name = split_name[0] + "_" + quantize_type + "." + split_name[1]

    # 重みファイルダウンロード
    model_path: str = os.path.join("model", model_name)
    if not os.path.exists(model_path):
        download_file(url=model_url[model], save_path=model_path)

    # カメラ準備

    picture_path = "./img/image.png"

    # HandLandmarker生成
    base_options: python.BaseOptions = python.BaseOptions(model_asset_path=model_path)
    options: vision.HandLandmarkerOptions = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
    )
    detector: vision.HandLandmarker = vision.HandLandmarker.create_from_options(
        options
    )  # type:ignore

    # FPS計測モジュール
    cvFpsCalc: CvFpsCalc = CvFpsCalc(buffer_len=10)
# World座標プロット
    if use_world_landmark:
        fig = plt.figure()
        r_ax = fig.add_subplot(121, projection="3d")  # type:ignore
        l_ax = fig.add_subplot(122, projection="3d")  # type:ignore
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    display_fps: float = cvFpsCalc.get()

    # カメラキャプチャ
    frame = cv2.imread(picture_path)

    #frameとしてはforsegment.mp4の10secの位置にある画像を使用
    # video_path = "data/forsegment.mp4"  # 動画のパス
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print("動画を読み込めませんでした。")
    #     return
    # # frameはforsegment.mp4の30secの位置にある画像を使用
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 600)  # 30秒の位置に移動
    # ret, frame = cap.read()
    # if not ret:
    #     print("フレームを読み込めませんでした。")
    #     return


    # 推論実施
    rgb_frame: mp.Image = mp.Image(
        image_format=mp.ImageFormat.SRGBA,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
    )
    detection_result: vision.HandLandmarkerResult = detector.detect(rgb_frame)

    # 外接矩形計算
    bboxes: List[List[int]] = calc_bounding_rect(frame, detection_result)

    # 描画
    debug_image: np.ndarray = copy.deepcopy(frame)
    debug_image = draw_debug(
        debug_image,
        detection_result,
        bboxes,
        display_fps,
    )

    # 描画（ワールド座標）
    if use_world_landmark:
        draw_world_landmarks(plt, [r_ax, l_ax], detection_result)

    # NOTE: segment anything
    # モデルの初期化
    sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # # 画像の読み込み

    # # 予測器に画像を設定
    predictor.set_image(frame)
    height, width, _ = frame.shape

    input_box = np.array([0, 0, width, height])

    brown_lower = np.array([10, 100, 100])  # HSV
    brown_upper = np.array([30, 200, 200])  # HSV

    white_lower = np.array([15, 0, 230])  # HSV
    white_upper = np.array([40, 50, 255])  # HSV

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_brown = cv2.inRange(hsv, brown_lower, brown_upper)
    mask_white = cv2.inRange(hsv, white_lower, white_upper)

    contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_image: np.ndarray = copy.deepcopy(frame)
    mask_image2: np.ndarray = copy.deepcopy(frame)

    # cv2.drawContours(mask_image, contours, -1, (0, 255, 0), 2)
    # cv2.drawContours(mask_image2, contours2, -1, (0, 255, 0), 2)

    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # cv2.imwrite(f"img/mask_image_{timestamp}.jpg", mask_brown)
    # cv2.imwrite(f"img/mask_image2_{timestamp}.jpg", mask_white)

    points = []
    point_coords = np.array([])
    point_labels = np.array([])

    # brownの部分について
    # y座標を、右手のbboxのy座標の最大値-100に設定し、それ以上のy座標の点を取得
    tmp_min_y = bboxes[0][1] +100

    # x座標は、右手のbboxのxmax+50に設定し、その範囲の点を取得
    tmp_min_x = bboxes[0][2] + 50
    for contour in contours_brown:
        for y in range(frame.shape[0] // 3, tmp_min_y, 16):
            for x in range(tmp_min_x, frame.shape[1], 16):
                if cv2.pointPolygonTest(contour, (x, y), False) == 1:
                    #ただし、左手のbbox領域内は除外
                    if bboxes[1][0] - 10 < x < bboxes[1][2] + 10 and bboxes[1][1]-10 < y < bboxes[1][3]+10:
                        continue
                    points.append([x, y])
                    point_labels = np.append(point_labels, 1)
                    break

    # whiteの部分について
    # x座標を、右手のbboxのxの最小値～画面の中央に設定し、その範囲の点を取得
    tmp_min_x = bboxes[0][0]
    # y座標は、右手のbboxのy座標の最大値を最大値とする
    tmp_max_y = bboxes[0][1]

    for contour in contours_white:
        for x in range(tmp_min_x, int(width*0.5), 8):
            for y in range(tmp_max_y, frame.shape[0], 8):
                if cv2.pointPolygonTest(contour, (x, y), False) == 1:
                    #ただし、左手のbbox領域内は除外
                    if bboxes[1][0] < x < bboxes[1][2] and bboxes[1][1] < y < bboxes[1][3]:
                        continue
                    # points.append([x, y])
                    # point_labels = np.append(point_labels, 0)


    # ポイントをNumPy配列に変換
    if len(points) > 0:
        point_coords = np.array(points)

    for i, (x, y) in enumerate(point_coords):
        if point_labels[i] == 1:
            cv2.circle(frame, (x, y), 5, (255, 0, 255), 2)
        else:
            cv2.circle(frame, (x, y), 5, (125, 125, 125), 2)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"img2/adjust-frame{timestamp}.jpg", frame)

    # 画像の前処理
    predictor.set_image(frame)

    # セグメンテーションの実行
    masks_segment, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=False,
    )

    mask_overlay = np.zeros_like(frame)
    mask_overlay[masks_segment[0] > 0] = [125, 255, 125]  # 緑色でマスクを表示
    cv2.imwrite(f"img3/mask_overlay_{timestamp}.jpg", mask_overlay)

    # ハフ変換で直線を検出
    gray_mask = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    #ハフ変換のlineのrho thetaを平均化
    if lines is not None:
        rho_values = []
        theta_values = []
        for rho, theta in lines[:, 0]:
            rho_values.append(rho)
            theta_values.append(theta)
        avg_rho = np.mean(rho_values)
        avg_theta = np.mean(theta_values)

        a = np.cos(avg_theta)
        b = np.sin(avg_theta)
        x0 = a * avg_rho
        y0 = b * avg_rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(mask_overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 赤色で描画
        cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 赤色で描画

        rho1 = avg_rho - (avg_rho - lines[0][0][0])
        x01 = a * rho1
        y01 = b * rho1
        x11 = int(x01 + 1000 * (-b))
        y11 = int(y01 + 1000 * (a))
        x21 = int(x01 - 1000 * (-b))
        y21 = int(y01 - 1000 * (a))
        cv2.line(mask_overlay, (x11, y11), (x21, y21), (255, 0, 0), 2)  # 青色で描画
        cv2.line(debug_image, (x11, y11), (x21, y21), (255, 0, 0), 2)



        rho2 = avg_rho + (avg_rho - lines[0][0][0])
        x02 = a * rho2
        y02 = b * rho2
        x12 = int(x02 + 1000 * (-b))
        y12 = int(y02 + 1000 * (a))
        x22 = int(x02 - 1000 * (-b))
        y22 = int(y02 - 1000 * (a))
        cv2.line(mask_overlay, (x12, y12), (x22, y22), (0, 255, 0), 2)  # 緑色で描画
        cv2.line(debug_image, (x12, y12), (x22, y22), (0, 255, 0), 2)



    cv2.imwrite(f"img3/mask_overlay2_{timestamp}.jpg", mask_overlay)

    # バウンディングボックスの取得
    y, x = np.where(masks_segment[0])
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    bbox_segment = [x_min, y_min, x_max, y_max]

    # バウンディングボックスの描画
    cv2.rectangle(debug_image, (bbox_segment[0], bbox_segment[1]), (bbox_segment[2], bbox_segment[3]), (0, 255, 0), 2)
    cv2.imwrite(f"img3/frame_with_bbox_{timestamp}.jpg", debug_image)





    # 画面反映
    # cv2.imshow("MediaPipe Hand Landmarks Detection Demo", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calc_bounding_rect(
    image: np.ndarray, detection_result: vision.HandLandmarkerResult
) -> List[List[int]]:
    image_width, image_height = image.shape[1], image.shape[0]

    bboxes: List[List[int]] = []
    for hand_landmarks in detection_result.hand_landmarks:
        landmark_array: np.ndarray = np.empty((0, 2), int)
        for landmark in hand_landmarks:
            landmark_x: int = min(int(landmark.x * image_width), image_width - 1)
            landmark_y: int = min(int(landmark.y * image_height), image_height - 1)

            landmark_point: np.ndarray = np.array((landmark_x, landmark_y))

            landmark_array = np.append(landmark_array, [landmark_point], axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        bboxes.append([x, y, x + w, y + h])

    return bboxes


def draw_debug(
    image: np.ndarray,
    detection_result: vision.HandLandmarkerResult,  # type:ignore
    bboxes: List[List[int]],
    display_fps: float,
) -> np.ndarray:
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_draw_info: Dict[
        int,
        Dict[str, Union[str, Tuple[int, int, int]]],
    ] = {
        0: {"name": "WRIST", "color": (0, 255, 0)},  # 手首  # 緑
        1: {  # 親指の手根中手関節（CM関節）
            "name": "THUMB_CMC",
            "color": (255, 0, 0),  # 赤
        },
        2: {  # 親指の中手指節関節（MP関節）
            "name": "THUMB_MCP",
            "color": (0, 0, 255),  # 青
        },
        3: {  # 親指の指節間関節（IP関節）
            "name": "THUMB_IP",
            "color": (255, 255, 0),  # 黄
        },
        4: {"name": "THUMB_TIP", "color": (0, 255, 255)},  # 親指の指先  # シアン
        5: {  # 人差し指の中手指節関節（MP関節）
            "name": "INDEX_FINGER_MCP",
            "color": (255, 0, 255),  # マゼンタ
        },
        6: {  # 人差し指の近位指節間関節（PIP関節）
            "name": "INDEX_FINGER_PIP",
            "color": (128, 128, 128),  # グレー
        },
        7: {  # 人差し指の遠位指節間関節（DIP関節）
            "name": "INDEX_FINGER_DIP",
            "color": (255, 128, 0),  # オレンジ
        },
        8: {"name": "INDEX_FINGER_TIP", "color": (128, 0, 255)},  # 人差し指の指先  # 紫
        9: {  # 中指の中手指節関節（MP関節）
            "name": "MIDDLE_FINGER_MCP",
            "color": (0, 128, 255),  # ライトブルー
        },
        10: {  # 中指の近位指節間関節（PIP関節）
            "name": "MIDDLE_FINGER_PIP",
            "color": (128, 255, 0),  # ライム
        },
        11: {  # 中指の遠位指節間関節（DIP関節）
            "name": "MIDDLE_FINGER_DIP",
            "color": (255, 128, 128),  # ライトレッド
        },
        12: {  # 中指の指先
            "name": "MIDDLE_FINGER_TIP",
            "color": (128, 128, 0),  # オリーブ
        },
        13: {  # 薬指の中手指節関節（MP関節）
            "name": "RING_FINGER_MCP",
            "color": (0, 128, 128),  # ティール
        },
        14: {  # 薬指の近位指節間関節（PIP関節）
            "name": "RING_FINGER_PIP",
            "color": (128, 0, 128),  # マルーン
        },
        15: {  # 薬指の遠位指節間関節（DIP関節）
            "name": "RING_FINGER_DIP",
            "color": (64, 64, 64),  # ダークグレー
        },
        16: {  # 薬指の指先
            "name": "RING_FINGER_TIP",
            "color": (192, 192, 192),  # シルバー
        },
        17: {  # 小指の中手指節関節（MP関節）
            "name": "PINKY_MCP",
            "color": (255, 69, 0),  # レッドオレンジ
        },
        18: {  # 小指の近位指節間関節（PIP関節）
            "name": "PINKY_PIP",
            "color": (75, 0, 130),  # インディゴ
        },
        19: {  # 小指の遠位指節間関節（DIP関節）
            "name": "PINKY_DIP",
            "color": (173, 255, 47),  # グリーンイエロー
        },
        20: {"name": "PINKY_TIP", "color": (220, 20, 60)},  # 小指の指先  # クリムゾン
    }

    line_info_list: List[List[int]] = [
        [0, 1],  # 手首から親指の手根中手関節（CM関節）
        [1, 2],  # 親指の手根中手関節（CM関節）から親指の中手指節関節（MP関節）
        [2, 3],  # 親指の中手指節関節（MP関節）から親指の指節間関節（IP関節）
        [3, 4],  # 親指の指節間関節（IP関節）から親指の指先
        [0, 5],  # 手首から人差し指の中手指節関節（MP関節）
        [
            5,
            6,
        ],  # 人差し指の中手指節関節（MP関節）から人差し指の近位指節間関節（PIP関節）
        [
            6,
            7,
        ],  # 人差し指の近位指節間関節（PIP関節）から人差し指の遠位指節間関節（DIP関節）
        [7, 8],  # 人差し指の遠位指節間関節（DIP関節）から人差し指の指先
        [0, 9],  # 手首から中指の中手指節関節（MP関節）
        [9, 10],  # 中指の中手指節関節（MP関節）から中指の近位指節間関節（PIP関節）
        [10, 11],  # 中指の近位指節間関節（PIP関節）から中指の遠位指節間関節（DIP関節）
        [11, 12],  # 中指の遠位指節間関節（DIP関節）から中指の指先
        [0, 13],  # 手首から薬指の中手指節関節（MP関節）
        [13, 14],  # 薬指の中手指節関節（MP関節）から薬指の近位指節間関節（PIP関節）
        [14, 15],  # 薬指の近位指節間関節（PIP関節）から薬指の遠位指節間関節（DIP関節）
        [15, 16],  # 薬指の遠位指節間関節（DIP関節）から薬指の指先
        [0, 17],  # 手首から小指の中手指節関節（MP関節）
        [17, 18],  # 小指の中手指節関節（MP関節）から小指の近位指節間関節（PIP関節）
        [18, 19],  # 小指の近位指節間関節（PIP関節）から小指の遠位指節間関節（DIP関節）
        [19, 20],  # 小指の遠位指節間関節（DIP関節）から小指の指先
    ]

    for handedness, hand_landmarks, _, bbox in zip(
        detection_result.handedness,
        detection_result.hand_landmarks,
        detection_result.hand_world_landmarks,
        bboxes,
    ):
        # 各ランドマーク情報整理
        landmark_dict: Dict[int, List[Union[int, float]]] = {}
        for index, landmark in enumerate(hand_landmarks):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_x: int = min(int(landmark.x * image_width), image_width - 1)
            landmark_y: int = min(int(landmark.y * image_height), image_height - 1)
            landmark_dict[index] = [landmark_x, landmark_y, landmark.z]

        # 接続線描画
        for line_info in line_info_list:
            cv2.line(
                image,
                tuple(landmark_dict[line_info[0]][:2]),
                tuple(landmark_dict[line_info[1]][:2]),
                (220, 220, 220),
                3,
                cv2.LINE_AA,
            )  # type:ignore

        # 各ランドマーク描画
        for index, landmark in landmark_dict.items():
            cv2.circle(
                image,
                (landmark[0], landmark[1]),
                5,
                landmark_draw_info[index]["color"],
                -1,
                cv2.LINE_AA,
            )  # type:ignore

        # 外接矩形描画
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # 左右描画
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

    # FPS
    cv2.putText(
        image,
        "FPS:" + str(display_fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return image


def draw_world_landmarks(
    plt: Any,
    ax_list: List[Any],
    detection_result: vision.HandLandmarkerResult,
) -> None:
    ax_list[0].cla()
    ax_list[0].set_xlim3d(-0.1, 0.1)
    ax_list[0].set_ylim3d(-0.1, 0.1)
    ax_list[0].set_zlim3d(-0.1, 0.1)
    ax_list[1].cla()
    ax_list[1].set_xlim3d(-0.1, 0.1)
    ax_list[1].set_ylim3d(-0.1, 0.1)
    ax_list[1].set_zlim3d(-0.1, 0.1)

    for handedness, _, hand_world_landmarks in zip(
        detection_result.handedness,
        detection_result.hand_landmarks,
        detection_result.hand_world_landmarks,
    ):
        handedness_index: int = 0
        if handedness[0].display_name == "Left":
            handedness_index = 0
        elif handedness[0].display_name == "Right":
            handedness_index = 1

        # 各ランドマーク情報整理
        landmark_dict: Dict[int, List[float]] = {}
        for index, landmark in enumerate(hand_world_landmarks):
            landmark_dict[index] = [landmark.x, landmark.y, landmark.z]

        palm_list: List[int] = [0, 1, 5, 9, 13, 17, 0]
        thumb_list: List[int] = [1, 2, 3, 4]
        index_finger_list: List[int] = [5, 6, 7, 8]
        middle_finger_list: List[int] = [9, 10, 11, 12]
        ring_finger_list: List[int] = [13, 14, 15, 16]
        pinky_list: List[int] = [17, 18, 19, 20]

        # 掌
        palm_x, palm_y, palm_z = [], [], []
        for index in palm_list:
            point = landmark_dict[index]
            palm_x.append(point[0])
            palm_y.append(point[2])
            palm_z.append(point[1] * (-1))

        # 親指
        thumb_x, thumb_y, thumb_z = [], [], []
        for index in thumb_list:
            point = landmark_dict[index]
            thumb_x.append(point[0])
            thumb_y.append(point[2])
            thumb_z.append(point[1] * (-1))

        # 人差し指
        index_finger_x, index_finger_y, index_finger_z = [], [], []
        for index in index_finger_list:
            point = landmark_dict[index]
            index_finger_x.append(point[0])
            index_finger_y.append(point[2])
            index_finger_z.append(point[1] * (-1))

        # 中指
        middle_finger_x, middle_finger_y, middle_finger_z = [], [], []
        for index in middle_finger_list:
            point = landmark_dict[index]
            middle_finger_x.append(point[0])
            middle_finger_y.append(point[2])
            middle_finger_z.append(point[1] * (-1))

        # 薬指
        ring_finger_x, ring_finger_y, ring_finger_z = [], [], []
        for index in ring_finger_list:
            point = landmark_dict[index]
            ring_finger_x.append(point[0])
            ring_finger_y.append(point[2])
            ring_finger_z.append(point[1] * (-1))

        # 小指
        pinky_x, pinky_y, pinky_z = [], [], []
        for index in pinky_list:
            point = landmark_dict[index]
            pinky_x.append(point[0])
            pinky_y.append(point[2])
            pinky_z.append(point[1] * (-1))

        ax_list[handedness_index].plot(palm_x, palm_y, palm_z)
        ax_list[handedness_index].plot(thumb_x, thumb_y, thumb_z)
        ax_list[handedness_index].plot(index_finger_x, index_finger_y, index_finger_z)
        ax_list[handedness_index].plot(
            middle_finger_x, middle_finger_y, middle_finger_z
        )
        ax_list[handedness_index].plot(ring_finger_x, ring_finger_y, ring_finger_z)
        ax_list[handedness_index].plot(pinky_x, pinky_y, pinky_z)

    plt.pause(0.001)

    return


if __name__ == "__main__":



    main()