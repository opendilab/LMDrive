import os
import json
import sys

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

verbose= False

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


def process(route):
    frames = len(os.listdir(os.path.join(route, "measurements")))
    for folder in dt:
        temp = dt[folder]
        try:
            files = os.listdir(os.path.join(route, folder))
        except:
            continue
        if frames != len(files):
            print('The folder (%s) has different frames (%d) with the record (%d).' % (os.path.join(route, folder), frames, len(files)))
        fs = []
        for file in files:
            fs.append(int(file.split('.')[0]))
        fs.sort()
        for i in range(len(fs)):
            if i == fs[i]:
                continue
            try:
                if verbose:
                    print(
                        "source: "+ os.path.join(route, folder, temp % fs[i]),
                        "target: "+ os.path.join(route, folder, temp % i),
                    )
                os.rename(
                    os.path.join(route, folder, temp % fs[i]),
                    os.path.join(route, folder, temp % i),
                )
            except Exception as e:
                print(e)


if __name__ == "__main__":
    dataset_root = sys.argv[1]
    list_file = os.path.join(dataset_root, 'blocked_stat.txt')
    routes = []

    with open(list_file, 'r') as f:
        for line in f.readlines():
            routes.append(line.strip().split()[0])

    routes = list(set(routes))
    with Pool(4) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))
