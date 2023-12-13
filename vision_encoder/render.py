import numpy as np
import cv2
import math

reweight_array = np.array([1.0, 3.5, 3.5, 1.0, 1.0, 3.5, 2.0, 8.0])


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


def convert_grid_to_xy(i, j):
    x = j - 24.5
    y = 29.5 - i
    return x, y


def find_peak_box(data):
    det_data = np.zeros((52, 52, 8))
    det_data[1:51, 1:51] = data
    res = []
    for i in range(1, 51):
        for j in range(1, 51):
            if det_data[i, j, 0] > 0.9 or (
                det_data[i, j, 0] > 0.4
                and det_data[i, j, 0] > det_data[i, j - 1, 0]
                and det_data[i, j, 0] > det_data[i, j + 1, 0]
                and det_data[i, j, 0] > det_data[i - 1, j, 0]
                and det_data[i, j, 0] > det_data[i + 1, j, 0]
            ):
                res.append((i - 1, j - 1))
    return res


def render_self_car(loc, ori, box, pixels_per_meter=5, max_distance=30):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    color = np.array([1, 1, 1])
    new_img = add_rect(img, loc, ori, box, 255, pixels_per_meter, max_distance, color)
    return new_img


def render(det_data, pixels_per_meter=5, max_distance=30, t=0):
    det_data = det_data * reweight_array
    box_ids = find_peak_box(det_data)
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    for poi in box_ids:
        i, j = poi
        center_x, center_y = convert_grid_to_xy(i, j)
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        #theta = det_data[i, j, 3] * np.pi
        #ori = np.array([math.cos(theta), math.sin(theta)])
        ori = np.array([det_data[i, j, 3], det_data[i, j, 4]])
        loc_x = center_x + det_data[i, j, 1] + t * det_data[i, j, 7] * ori[0]
        loc_y = center_y + det_data[i, j, 2] - t * det_data[i, j, 7] * ori[1]
        loc = np.array([loc_x, -loc_y])
        box = np.array(det_data[i, j, 5:7])
        if box[0] < 1.5:
            box = box * 1.5
        color = np.array([1, 1, 1])
        new_img = add_rect(
            act_img, loc, ori, box, 255, pixels_per_meter, max_distance, color
        )
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)[:, :, 0]
    img = img.astype(np.uint8)
    return img


def render_waypoints(waypoints, pixels_per_meter=5, max_distance=30):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size), np.uint8)
    for i in range(len(waypoints)):
        new_loc = waypoints[i]
        new_loc = new_loc * pixels_per_meter + pixels_per_meter * max_distance
        new_loc = np.around(new_loc)
        new_loc = tuple(new_loc.astype(int))
        img = cv2.circle(img, new_loc, 3, 255, -1)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img
