"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process


from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.carla_dataset_llm import CarlaVoiceDataset

@registry.register_builder("carla_voice")
class CarlaDatasetBuilder(BaseDatasetBuilder):
    DATASET_CONFIG_DICT = {"default": "configs/datasets/carla/defaults.yaml"}
    def __init__(self, cfg=None):
        #super().__init__()
        self.config = cfg

    def build_datasets(self):
        datasets = self.build()
        return datasets

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        #self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # annotation path
            ann_path = ann_info.get(split).storage
            towns = ann_info.get(split).towns
            weathers = ann_info.get(split).weathers
            scale = ann_info.get(split).scale
            enable_start_frame_augment = ann_info.get(split).enable_start_frame_augment
            token_max_length = ann_info.get(split).token_max_length
            enable_notice = ann_info.get(split).enable_notice

            # create datasets
            datasets[split] = CarlaVoiceDataset(
                dataset_root=ann_path,
                towns=towns,
                weathers=weathers,
                scale=scale,
                enable_start_frame_augment=enable_start_frame_augment,
                token_max_length=token_max_length,
                enable_notice=enable_notice,
            )

        return datasets

