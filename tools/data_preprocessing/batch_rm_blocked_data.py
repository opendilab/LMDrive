import os
import json
import sys

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

dt = {}
dt["topdown"] = "%04d.jpg"
dt["rgb_right"] = "%04d.jpg"
dt["rgb_left"] = "%04d.jpg"
dt["rgb_front"] = "%04d.jpg"
dt["rgb_rear"] = "%04d.jpg"
dt["measurements"] = "%04d.json"
dt["lidar"] = "%04d.npy"
dt["lidar_odd"] = "%04d.npy"
dt["birdview"] = "%04d.jpg"
dt["affordances"] = "%04d.npy"
dt["actors_data"] = "%04d.json"
dt["3d_bbs"] = "%04d.npy"
dt['rgb_full'] = "%04d.jpg"
dt['measurements_full'] = '%04d.json'
#3d_bbs  actors_data  affordances  birdview  lidar  lidar_odd  measurements  rgb_front  rgb_left  rgb_rear  rgb_right  topdown
#3d_bbs  actors_data  affordances  birdview  lidar  lidar_odd  measurements  measurements_full  rgb_full  topdown


def process(task):
    route_dir, end_id, length = task
    for i in range(end_id - length + 6, end_id - 3):
        for key in dt:
            try:
                os.remove(os.path.join(route_dir, key, dt[key] % i))
            except Exception as e:
                pass


if __name__ == "__main__":
    dataset_root = sys.argv[1]
    list_file = os.path.join(dataset_root, 'blocked_stat.txt')
    tasks = []
    for line in open(list_file, "r").readlines():
        line = line.strip().split()
        tasks.append([line[0], int(line[1]), int(line[2])])
    with Pool(8) as p:
        r = list(tqdm(p.imap(process, tasks), total=len(tasks)))
