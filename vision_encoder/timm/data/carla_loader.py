""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch.utils.data
import numpy as np

from .transforms_carla_factory import (
    create_carla_rgb_transform,
    create_carla_seg_transform,
)
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .distributed_sampler import OrderedDistributedSampler


def create_carla_loader(
    dataset,
    input_size,
    batch_size,
    multi_view_input_size=None,
    is_training=False,
    no_aug=False,
    scale=[1.0, 1.0],
    color_jitter=0.4,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    num_workers=1,
    distributed=False,
    collate_fn=None,
    pin_memory=False,
    fp16=False,
    persistent_workers=True,
):
    dataset.rgb_transform = create_carla_rgb_transform(
        input_size,
        is_training=is_training,
        scale=scale,
        color_jitter=color_jitter,
        interpolation=interpolation,
        mean=mean,
        std=std,
    )

    dataset.rgb_center_transform = create_carla_rgb_transform(
        128,
        is_training=is_training,
        scale=None,
        color_jitter=color_jitter,
        interpolation=interpolation,
        mean=mean,
        std=std,
        need_scale=False,
    )

    dataset.seg_transform = create_carla_seg_transform(
        input_size,
        is_training=is_training,
        scale=scale,
        color_jitter=0,
        interpolation=interpolation,
        mean=mean,
        std=std,
    )

    dataset.depth_transform = create_carla_rgb_transform(
        input_size,
        is_training=is_training,
        scale=scale,
        color_jitter=0,
        interpolation=interpolation,
        mean=mean,
        std=std,
    )
    if multi_view_input_size is None:
        multi_view_input_size = input_size
    dataset.multi_view_transform = create_carla_rgb_transform(
        multi_view_input_size,
        is_training=is_training,
        scale=scale,
        color_jitter=color_jitter,
        interpolation=interpolation,
        mean=mean,
        std=std,
    )

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset)
        and sampler is None
        and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        persistent_workers=persistent_workers,
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop("persistent_workers")  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    return loader
