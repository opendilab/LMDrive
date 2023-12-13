import os
import sys
import json

from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

'''
merge measurements from all frames into one file
'''

def process(route):
    try:
        frames = len(list(os.listdir(os.path.join(route, "measurements_full"))))
        measurements = []
        for i in range(frames):
            measurements.append(json.load(
                open(os.path.join(route, "measurements/%04d.json" % i), 'r')
            ))
        json.dump(measurements, open(os.path.join(route, "measurements_all.json"), 'w'))

    except Exception as e:
        print(e)
        print('The folder %s has an existing problem' % route)


if __name__ == "__main__":
    dataset_root = sys.argv[1]
    list_file = os.path.join(dataset_root, 'dataset_index.txt')
    routes = []
    for line in open(list_file, "r").readlines():
        path = line.split()[0].strip()
        routes.append(os.path.join(dataset_root, path))
    with Pool(8) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))
