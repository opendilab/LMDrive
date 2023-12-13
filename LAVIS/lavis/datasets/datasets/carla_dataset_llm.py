import os
import random
import copy
import re
import logging
from pathlib import Path

import json
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

from .base_io_dataset import BaseIODataset
from .transforms_carla_factory import create_carla_rgb_transform

curr_dir = Path(__file__).parent
instruction_json = os.path.join(curr_dir, "../../../../", "leaderboard/leaderboard/envs", 'instruction_dict.json')
INSTRUCTION_DICT = json.load(open(instruction_json))


_logger = logging.getLogger(__name__)

def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw


def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [ np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0,0,1,0],
        [0,0,0,1]
    ]

def lidar_to_raw_features(lidar):
    def preprocess(lidar_xyzr, lidar_painted=None):

        idx = (lidar_xyzr[:,0] > -1.2)&(lidar_xyzr[:,0] < 1.2)&(lidar_xyzr[:,1]>-1.2)&(lidar_xyzr[:,1]<1.2)

        idx = np.argwhere(idx)

        if lidar_painted is None:
            return np.delete(lidar_xyzr, idx, axis=0)
        else:
            return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

    lidar_xyzr = preprocess(lidar)

    idxs = np.arange(len(lidar_xyzr))
    np.random.shuffle(idxs)
    lidar_xyzr = lidar_xyzr[idxs]

    lidar = np.zeros((40000, 4), dtype=np.float32)
    num_points = min(40000, len(lidar_xyzr))
    lidar[:num_points,:4] = lidar_xyzr[:num_points]
    lidar[np.isinf(lidar)] = 0
    lidar[np.isnan(lidar)] = 0
    lidar = rotate_lidar(lidar, -90).astype(np.float32)
    return lidar, num_points

def check_data(data, info):
    for key in data:
        if isinstance(data[key], np.ndarray):
            if np.isnan(data[key]).any():
                print(key)
                print(info)
                data[key][np.isnan(data[key])] = 0
            if np.isinf(data[key]).any():
                print(key)
                print(info)
                data[key][np.isinf(data[key])] = 0
        elif isinstance(data[key], torch.Tensor):
            if torch.isnan(data[key]).any():
                print(key)
                print(info)
                data[key][torch.isnan(data[key])] = 0
            if torch.isinf(data[key]).any():
                print(key)
                print(info)
                data[key][torch.isinf(data[key])] = 0
    return data


class CarlaVoiceDataset(BaseIODataset):
    def __init__(
        self,
        dataset_root,
        towns=None,
        weathers=None,
        scale=None,
        is_training=False,
        input_rgb_size=224,
        input_multi_view_size=128,
        input_lidar_size=224,
        token_max_length=32,
        sample_interval=2,
        enable_start_frame_augment=False,
        enable_notice=False,
        **kwargs,
    ):
        super().__init__()

        self.token_max_length = token_max_length
        self.rgb_transform = create_carla_rgb_transform(
            input_rgb_size,
            is_training=is_training,
            scale=scale,
        )
        self.rgb_center_transform = create_carla_rgb_transform(
            128,
            scale=None,
            is_training=is_training,
            need_scale=False,
        )
        self.multi_view_transform = create_carla_rgb_transform(
            input_multi_view_size,
            scale=scale,
            is_training=is_training,
        )

        self.scenario_infos = self._get_scenario_paths(dataset_root, weathers, towns)
        _logger.info("Scenario nums: %d" % len(self.scenario_infos))
        self.instruction_dict = INSTRUCTION_DICT
        self.sample_interval = sample_interval
        self.enable_start_frame_augment = enable_start_frame_augment
        self.enable_notice = enable_notice
        if self.enable_notice:
            raw_notice_data = self._load_json(os.path.join(dataset_root, 'notice_instruction_list.json'))
            self.notice_data = {}
            for key in raw_notice_data:
                self.notice_data[os.path.join(dataset_root, key)] = raw_notice_data[key]

    def collater(self, samples):
        return default_collate(samples)

    def _get_scenario_paths(self, dataset_root, weathers, towns):
        scenario_infos = []
        dataset_indexs = self._load_text(os.path.join(dataset_root, 'navigation_instruction_list.txt')).split('\n')
        for line in dataset_indexs:
            if len(line) < 10: continue
            info = json.loads(line.strip())
            # result {dict}: route_path, town_id, weather_id, start_frame, end_frame, instruction, instruction_id, instruction_args, route_frames
            if towns is not None:
                if info['town_id'] not in towns:
                    continue
            if weathers is not None:
                if info['weather_id'] not in weathers:
                    continue
            info['route_path'] = os.path.join(dataset_root, info['route_path'])
            scenario_infos.append(info)
        return scenario_infos

    def __len__(self):
        return len(self.scenario_infos)

    def pad_and_stack(self, data):
        if isinstance(data[0], np.ndarray):
            for _ in range(self.token_max_length - len(data)):
                data.append(np.zeros_like(data[0]))
            data = np.stack(data, 0)
        elif torch.is_tensor(data[0]):
            for _ in range(self.token_max_length - len(data)):
                data.append(torch.zeros_like(data[0]))
            data = torch.stack(data, 0)
        else:
            for _ in range(self.token_max_length - len(data)):
                data.append(0)
            data = np.array(data).reshape(-1)
        return data

    def __getitem__(self, idx):
        info = self.scenario_infos[idx]
        route_path = info['route_path']
        route_frames = int(info['route_frames'])
        town_id = info['town_id']
        weather_id = info['weather_id']


        if 'Turn' in info['instruction']:
            info['end_frame'] = min(route_frames - 1, info['end_frame'] + 12)

        sample_interval = self.sample_interval

        start_frame_id = info['start_frame'] + random.randint(0, sample_interval)
        if self.enable_start_frame_augment and len(info['instruction_args']) == 0: # if instruction_args has no values, it means the instruction doesn't include distance
            if 'Other' not in info['instruction'] or info['instruction'] != 'Follow-01':
                augment_range = min(16, max(0, self.token_max_length * self.sample_interval - (info['end_frame'] - info['start_frame'])))
                start_frame_id = max(0, start_frame_id - random.randint(0, augment_range))
        end_frame_id = min(info['end_frame'] + 1, start_frame_id + self.token_max_length * sample_interval - random.randint(0, sample_interval-1))

        # we construct notice data after obtaining the final start/end frame id
        if self.enable_notice:
            notice_frame_id = []
            notice_text = []
            if route_path in self.notice_data:
                notice_list = self.notice_data[route_path]
                notice_list = [x for x in notice_list if x['frame_id'] > start_frame_id and x['frame_id'] < end_frame_id - 16]
                if len(notice_list) < 1 or random.random() < 0.75:
                    notice_frame_id = -1
                    notice_text = ''
                else:
                    notice = random.choice(notice_list)
                    # we convert the abslote poisition to the relative position
                    notice_frame_id = (notice['frame_id'] - start_frame_id) // sample_interval + 1
                    notice_text = np.random.choice(self.instruction_dict[str(notice['instruction_id'])])
            else:
                notice_frame_id = -1
                notice_text = ''

        measurements = self._load_json(os.path.join(route_path, "measurements_all.json"))
        ego_theta = measurements[start_frame_id]['theta']
        processed_data = {}

        if np.isnan(ego_theta):
            ego_theta = 0
        R = np.array(
            [[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]])
        origin_x = measurements[start_frame_id]['gps_x']
        origin_y = measurements[start_frame_id]['gps_y']


        ego_throttles = []
        ego_steers = []
        ego_brakes = []
        ego_velocitys = []
        ego_xs = []
        ego_ys = []
        local_positions = []
        local_future_waypoints = []
        text_before_img = []
        text_after_img = []
        target_points = []

        for frame_id in range(start_frame_id, end_frame_id, sample_interval):
            ego_x = measurements[frame_id]['gps_x']
            ego_y = measurements[frame_id]['gps_y']
            velocity = measurements[frame_id]['speed']
            local_position = np.array([ego_x - origin_x, ego_y - origin_y])
            local_position = R.T.dot(local_position)
            text_before_img.append('<frame %.1f,%.1f>' % (local_position[0], local_position[1]))
            # text_before_img.append('<frame=%d;x=%.1f;y=%.1f;speed=%.1f>' % (frame_id-start_frame_id, local_position[0], local_position[1], velocity))
            text_after_img.append('</frame>')

            ego_xs.append(ego_x)
            ego_ys.append(ego_y)
            ego_throttles.append(measurements[frame_id]['throttle'])
            ego_steers.append(measurements[frame_id]['steer'])
            ego_brakes.append(int(measurements[frame_id]['brake']))
            ego_velocitys.append(velocity)
            local_positions.append(local_position.reshape(-1))
            local_ego_theta = measurements[frame_id]['theta']
            if np.isnan(local_ego_theta):
                local_ego_theta = 0
            local_R = np.array(
                [[np.cos(np.pi / 2 + local_ego_theta), -np.sin(np.pi / 2 + local_ego_theta)],
                [np.sin(np.pi / 2 + local_ego_theta), np.cos(np.pi / 2 + local_ego_theta)]])

            x_command = measurements[frame_id]["x_command"]
            y_command = measurements[frame_id]["y_command"]
            local_command_point = np.array([x_command - ego_x, y_command - ego_y])
            local_command_point = local_R.T.dot(local_command_point)
            if any(np.isnan(local_command_point)):
                local_command_point[np.isnan(local_command_point)] = np.mean(
                    local_command_point
                )
            local_command_point = local_command_point.reshape(-1)
            target_points.append(local_command_point)

            local_future_waypoints_temp = []
            for future_frame_delta in range(1, 6):
                future_frame_id = min(frame_id + future_frame_delta * 5, route_frames-1)
                future_ego_x = measurements[future_frame_id]['gps_x']
                future_ego_y = measurements[future_frame_id]['gps_y']
                future_waypoint = np.array([future_ego_x - ego_x, future_ego_y - ego_y])
                future_waypoint = local_R.T.dot(future_waypoint)
                local_future_waypoints_temp.append(future_waypoint.reshape(1, 2))
            local_future_waypoints.append(np.concatenate(local_future_waypoints_temp, axis=0).reshape(-1))

        valid_frames = len(ego_xs)
        ego_throttles = self.pad_and_stack(ego_throttles)
        ego_steers = self.pad_and_stack(ego_steers)
        ego_brakes = self.pad_and_stack(ego_brakes)
        ego_velocitys = self.pad_and_stack(ego_velocitys)
        ego_xs = self.pad_and_stack(ego_xs)
        ego_ys = self.pad_and_stack(ego_ys)
        target_points = self.pad_and_stack(target_points)
        local_positions = self.pad_and_stack(local_positions)
        local_future_waypoints = self.pad_and_stack(local_future_waypoints)

        lidar_data = []
        lidar_num_points = []
        rgb_front = []
        rgb_center = []
        rgb_left = []
        rgb_right = []
        rgb_rear = []

        for frame_id in range(start_frame_id, end_frame_id, sample_interval):
            sensor_data = self._extract_data_item(route_path, frame_id)
            lidar_data.append(sensor_data['lidar'])
            lidar_num_points.append(sensor_data['num_points'])
            rgb_front.append(sensor_data['rgb'])
            rgb_center.append(sensor_data['rgb_center'])
            rgb_left.append(sensor_data['rgb_left'])
            rgb_right.append(sensor_data['rgb_right'])
            rgb_rear.append(sensor_data['rgb_rear'])

        processed_data['lidar'] = self.pad_and_stack(lidar_data)
        processed_data['num_points'] = self.pad_and_stack(lidar_num_points)
        processed_data['rgb_front'] = self.pad_and_stack(rgb_front)
        processed_data['rgb_left'] = self.pad_and_stack(rgb_left)
        processed_data['rgb_right'] = self.pad_and_stack(rgb_right)
        processed_data['rgb_rear'] = self.pad_and_stack(rgb_rear)
        processed_data['rgb_center'] = self.pad_and_stack(rgb_center)

        instruction_text = np.random.choice(self.instruction_dict[str(info['instruction_id'])])
        try:
            if '[x]' in instruction_text:
                instruction_text.replace('[x]', str(info['instruction_args'][0]))
            if 'left/right' in instruction_text:
                instruction_text.replace('left/right', str(info['instruction_args'][1]))
            if '[y]' in instruction_text:
                instruction_text.replace('[y]', str(info['instruction_args'][2]))
        except Exception as e:
            _logger.error(e)
            _logger.info(info)
            _logger.info(instruction_text)

        processed_data['target_point'] = torch.from_numpy(target_points).float()
        processed_data['valid_frames'] = valid_frames
        processed_data['text_input'] = instruction_text
        processed_data['text_before_img'] = '|'.join(text_before_img)
        processed_data['text_after_img'] = '|'.join(text_after_img)
        processed_data['ego_throttles'] = ego_throttles
        processed_data['ego_steers'] = ego_steers
        processed_data['ego_brakes'] = ego_brakes
        processed_data['velocity'] = torch.from_numpy(np.array(ego_velocitys)).float()
        processed_data['local_positions'] = local_positions
        processed_data['local_future_waypoints'] = local_future_waypoints
        if self.enable_notice:
            processed_data['notice_frame_id'] = notice_frame_id
            processed_data['notice_text'] = notice_text

        return processed_data


    def _extract_data_item(self, route_path, frame_id):
        data = {}
        # You can use tools/data/batch_merge_data.py to generate FULL image (including front, left, right) for reducing io cost
        rgb_full_image = self._load_image(
            os.path.join(route_path, "rgb_full", "%04d.jpg" % frame_id)
        )
        rgb_image = rgb_full_image.crop((0, 0, 800, 600))
        rgb_left_image = rgb_full_image.crop((0, 600, 800, 1200))
        rgb_right_image = rgb_full_image.crop((0, 1200, 800, 1800))
        rgb_rear_image = rgb_full_image.crop((0, 1800, 800, 2400))

        '''
        rgb_image = self._load_image(
            os.path.join(route_path, "rgb_front", "%04d.jpg" % frame_id)
        )
        rgb_left_image = self._load_image(
            os.path.join(route_path, "rgb_left", "%04d.jpg" % frame_id)
        )
        rgb_right_image = self._load_image(
            os.path.join(route_path, "rgb_right", "%04d.jpg" % frame_id)
        )
        '''
        lidar_unprocessed_front = self._load_npy(
            os.path.join(route_path, "lidar", "%04d.npy" % frame_id)
        )[..., :4]
        lidar_unprocessed_back = self._load_npy(
            os.path.join(route_path, "lidar_odd", "%04d.npy" % max(frame_id - 1, 0))
        )[..., :4]
        lidar_unprocessed = np.concatenate([lidar_unprocessed_front, lidar_unprocessed_back])
        lidar_processed, num_points= lidar_to_raw_features(lidar_unprocessed)
        data['lidar'] = lidar_processed
        data['num_points'] = num_points

        if self.rgb_transform is not None:
            rgb_main_image = self.rgb_transform(rgb_image)
        data["rgb"] = rgb_main_image

        if self.rgb_center_transform is not None:
            rgb_center_image = self.rgb_center_transform(rgb_image)
        data["rgb_center"] = rgb_center_image

        if self.multi_view_transform is not None:
            rgb_left_image = self.multi_view_transform(rgb_left_image)
            rgb_right_image = self.multi_view_transform(rgb_right_image)
            rgb_rear_image = self.multi_view_transform(rgb_rear_image)
        data["rgb_left"] = rgb_left_image
        data["rgb_right"] = rgb_right_image
        data["rgb_rear"] = rgb_rear_image

        data = check_data(data, info=route_path+str(frame_id))
        return data
