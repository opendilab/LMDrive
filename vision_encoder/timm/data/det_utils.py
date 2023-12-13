import math
import json
import os
import copy

from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

from skimage.measure import block_reduce
from .heatmap_utils import generate_heatmap, get_yaw_angle


def convert_grid_to_xy(i, j):
    x = j - 24.5
    y = 29.5 - i
    return x, y


def generate_det_data(
    heatmap, measurements, actors_data, pixels_per_meter=5, max_distance=30, return_object_ids=False
):
    traffic_heatmap = block_reduce(heatmap, block_size=(5, 5), func=np.mean)
    traffic_heatmap = np.clip(traffic_heatmap, 0.0, 255.0)
    traffic_heatmap = traffic_heatmap[:50, 5:55]
    det_data = np.zeros((50, 50, 8))

    ego_x = measurements["x"]
    ego_y = measurements["y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    need_deleted_ids = []
    for _id in actors_data:
        raw_loc = actors_data[_id]["loc"]
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        new_loc[1] = -new_loc[1]
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        dis = new_loc[0] ** 2 + new_loc[1] ** 2
        if (
            dis <= 1
            or dis >= (max_distance + 3) ** 2 * 2
            or "box" not in actors_data[_id]
        ):
            need_deleted_ids.append(_id)
            continue
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])

    for _id in need_deleted_ids:
        del actors_data[_id]

    for i in range(50):  # Vertical
        for j in range(50):  # horizontal
            if traffic_heatmap[i][j] < 0.05 * 255.0:
                continue
            center_x, center_y = convert_grid_to_xy(i, j)
            min_dis = 1000
            min_id = None
            for _id in actors_data:
                loc = actors_data[_id]["loc"][:2]
                ori = actors_data[_id]["ori"][:2]
                box = actors_data[_id]["box"]
                dis = (loc[0] - center_x) ** 2 + (loc[1] - center_y) ** 2
                if dis < min_dis:
                    min_dis = dis
                    min_id = _id
            if min_id is None:
                det_data[i][j] = np.array(
                    [
                        0,0,0,0,0,0,0,0
                    ]
                )
                continue
            loc = actors_data[min_id]["loc"][:2]
            ori = actors_data[min_id]["ori"][:2]
            box = actors_data[min_id]["box"]
            # theta = (get_yaw_angle(ori) / np.pi + 2) % 2
            speed = np.linalg.norm(actors_data[min_id]["vel"])
            prob = np.power(0.5 / max(0.5, np.sqrt(min_dis) - np.sqrt(0.5)), 0.5)
            det_data[i][j] = np.array(
                [
                    prob,
                    (loc[0] - center_x) / 3.5,
                    (loc[1] - center_y) / 3.5,
                    ori[0],
                    ori[1],
                    box[0] / 3.5,
                    box[1] / 2.0,
                    speed / 8.0,
                ]
            )
    if return_object_ids:
        results = {}
        for _id in actors_data:
            min_dis = 0.5
            min_pos = None
            loc = actors_data[_id]["loc"][:2]
            for i in range(20):
                for j in range(20):
                    center_x, center_y = convert_grid_to_xy(i, j)
                    dis = (loc[0] - center_x) ** 2 + (loc[1] - center_y) ** 2
                    if dis < min_dis:
                        min_dis = dis
                        min_pos = (i, j)
            if min_pos is None:
                continue
            results[_id] = min_pos
        return det_data, results
    else:
        return det_data
