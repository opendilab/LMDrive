""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random
import torch
import numpy as np
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


class RandomResize:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, pil_img):
        img_size = pil_img.size
        r_scale = (self.scale[1] - self.scale[0]) * random.random() + self.scale[0]
        pil_img = pil_img.resize(
            (int(img_size[0] * r_scale), int(img_size[1] * r_scale))
        )
        return pil_img


class SegConvert:
    def __init__(self):
        pass

    def __call__(self, image):
        # https://carla.readthedocs.io/en/0.9.10/ref_sensors/#semantic-segmentation-camera
        data = np.array(image)
        data = data[:, :, 0]
        h, w = data.shape
        out = np.zeros((h, w)).astype(np.uint8)
        out[data == 4] = 1  # Pedestrian
        out[data == 6] = 2  # RoadLine
        out[data == 7] = 3  # Road
        out[data == 8] = 4  # SideWalk
        out[data == 10] = 1  # Vehicles
        out[data == 11] = 0
        out[data == 12] = 5  # Traffic Sign
        out[data == 13] = 6  # Sky
        out[data == 19] = 7  # Static
        out = torch.tensor(out).flatten(0).long()
        return out


def create_carla_rgb_transform(
    input_size,
    crop_size=None,
    use_prefetcher=False,
    scale=None,
    need_scale=True,
    interpolation="bilinear",
    is_training=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((340, 288)))
        else:
            tfl.append(
                Resize2FixedSize(
                    (int((input_size_num + 32) / 3.0 * 4.0), input_size_num + 32)
                )
            )
    if is_training:
        if scale:
            tfl.append(RandomResize(scale))
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


def create_carla_seg_transform(
    input_size,
    crop_size=None,
    is_training=False,
    use_prefetcher=False,
    scale=None,
    color_jitter=0.0,
    auto_augment=None,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if input_size_num == 112:
        tfl.append(Resize2FixedSize((170, 128)))
    elif input_size_num == 224:
        tfl.append(Resize2FixedSize((341, 256)))
    elif input_size_num == 256:
        tfl.append(Resize2FixedSize((288, 288)))
    else:
        raise ValueError("Can't find proper crop size")
    if is_training:
        if scale:
            tfl.append(RandomResize(scale))
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(SegConvert())

    return transforms.Compose(tfl)
