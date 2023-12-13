import os
import sys
import json
from multiprocessing import Pool

from tqdm import tqdm
from PIL import Image
import numpy as np


def process(route):
    try:
        frames = len(os.listdir(os.path.join(route, "measurements")))
        if not os.path.exists(os.path.join(route, "rgb_full")):
            os.mkdir(os.path.join(route, "rgb_full"))
        if not os.path.exists(os.path.join(route, "measurements_full")):
            os.mkdir(os.path.join(route, "measurements_full"))
        for i in range(frames):
            img_front = Image.open(os.path.join(route, "rgb_front/%04d.jpg" % i))
            img_left = Image.open(os.path.join(route, "rgb_left/%04d.jpg" % i))
            img_right = Image.open(os.path.join(route, "rgb_right/%04d.jpg" % i))
            img_rear = Image.open(os.path.join(route, "rgb_rear/%04d.jpg" % i))
            new = Image.new(img_front.mode, (800, 2400))
            new.paste(img_front, (0, 0))
            new.paste(img_left, (0, 600))
            new.paste(img_right, (0, 1200))
            new.paste(img_rear, (0, 1800))
            new.save(os.path.join(route, "rgb_full", "%04d.jpg" % i))

            measurements = json.load(
                open(os.path.join(route, "measurements/%04d.json" % i))
            )
            actors_data = json.load(
                open(os.path.join(route, "actors_data/%04d.json" % i))
            )
            affordances = np.load(
                os.path.join(route, "affordances/%04d.npy" % i), allow_pickle=True
            )

            measurements["actors_data"] = actors_data
            measurements["stop_sign"] = affordances.item()["stop_sign"]
            json.dump(
                measurements,
                open(os.path.join(route, "measurements_full/%04d.json" % i), "w"),
            )
    except Exception as e:
        print(e)
        print('The folder %s has an existing problem, and we will proceed to remove it...' % route)
        os.system('rm -rf %s' % route)


if __name__ == "__main__":
    dataset_root = sys.argv[1]
    list_file = os.path.join(dataset_root, 'dataset_index.txt')
    routes = []
    for line in open(list_file, "r").readlines():
        path = line.split()[0].strip()
        routes.append(os.path.join(dataset_root, path))
    with Pool(8) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))
