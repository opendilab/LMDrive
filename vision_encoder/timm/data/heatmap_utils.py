import math
import json
import os

from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np


# VALUES = [255, 150, 120, 90, 60, 30][::-1]
# EXTENT = [0, 0.2, 0.4, 0.6, 0.8, 1.0][::-1]


VALUES = [255]
EXTENT = [0]


def add_rect(img, loc, ori, box, value, pixels_per_meter, max_distance, color):
    img_size = max_distance * pixels_per_meter * 2
    vet_ori = np.array([-ori[1], ori[0]])
    hor_offset = box[0] * ori
    vet_offset = box[1] * vet_ori
    left_up = (loc + hor_offset + vet_offset + max_distance) * pixels_per_meter
    left_down = (loc + hor_offset - vet_offset + max_distance) * pixels_per_meter
    right_up = (loc - hor_offset + vet_offset + max_distance) * pixels_per_meter
    right_down = (loc - hor_offset - vet_offset + max_distance) * pixels_per_meter
    left_up = np.around(left_up).astype(int)
    left_down = np.around(left_down).astype(int)
    right_down = np.around(right_down).astype(int)
    right_up = np.around(right_up).astype(int)
    left_up = list(left_up)
    left_down = list(left_down)
    right_up = list(right_up)
    right_down = list(right_down)
    color = [int(x) for x in value * color]
    cv2.fillConvexPoly(img, np.array([left_up, left_down, right_down, right_up]), color)
    return img


def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw


def generate_future_waypoints(measurements, pixels_per_meter=5, max_distance=30):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size), np.uint8)
    ego_x = measurements["gps_x"]
    ego_y = measurements["gps_y"]
    ego_theta = measurements["theta"] + np.pi / 2
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    for waypoint in measurements["future_waypoints"]:
        new_loc = R.T.dot(np.array([waypoint[0] - ego_x, waypoint[1] - ego_y]))
        if new_loc[0] ** 2 + new_loc[1] ** 2 > (max_distance + 3) ** 2 * 2:
            break
        new_loc = new_loc * pixels_per_meter + pixels_per_meter * max_distance
        new_loc = np.around(new_loc)
        new_loc = tuple(new_loc.astype(int))
        img = cv2.circle(img, new_loc, 3, 255, -1)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def generate_heatmap(measurements, actors_data, pixels_per_meter=5, max_distance=30):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), int)
    ego_x = measurements["x"]
    ego_y = measurements["y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    ego_id = None
    for _id in actors_data:
        color = np.array([1, 1, 1])
        if actors_data[_id]["tpe"] == 2:
            if int(_id) == int(measurements["affected_light_id"]):
                if actors_data[_id]["sta"] == 0:
                    color = np.array([1, 1, 1])
                else:
                    color = np.array([0, 0, 0])
                yaw = get_yaw_angle(actors_data[_id]["ori"])
                TR = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
                actors_data[_id]["loc"] = np.array(
                    actors_data[_id]["loc"][:2]
                ) + TR.T.dot(np.array(actors_data[_id]["taigger_loc"])[:2])
                actors_data[_id]["ori"] = np.array(actors_data[_id]["ori"])
                actors_data[_id]["box"] = np.array(actors_data[_id]["trigger_box"]) * 2
            else:
                continue
        raw_loc = actors_data[_id]["loc"]
        if (raw_loc[0] - ego_x) ** 2 + (raw_loc[1] - ego_y) ** 2 <= 1:
            ego_id = _id
            color = np.array([0, 1, 1])
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])
        if int(_id) in measurements["is_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_bike_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_junction_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_pedestrian_present"]:
            color = np.array([1, 1, 1])
        actors_data[_id]["color"] = color

    if ego_id is not None and ego_id in actors_data:
        del actors_data[ego_id]  # Do not show ego car
    for _id in actors_data:
        if actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
            if int(_id) != int(measurements["affected_light_id"]):
                continue
            if actors_data[_id]["sta"] != 0:
                continue
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        if box[0] < 1.5:
            box = box * (1.5 / box[0])  # FIXME enlarge the size of pedstrian and bike
        color = actors_data[_id]["color"]
        for i in range(len(VALUES)):
            act_img = add_rect(
                act_img,
                loc,
                ori,
                box + EXTENT[i],
                VALUES[i],
                pixels_per_meter,
                max_distance,
                color,
            )
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = img[:, :, 0]
    return img
