import os
import sys
import json

from tqdm import tqdm
from multiprocessing import Pool
import numpy as np


def process(route):
    try:
        os.system('rm -rf %s' % (os.path.join(route, 'rgb_front')))
        os.system('rm -rf %s' % (os.path.join(route, 'rgb_left')))
        os.system('rm -rf %s' % (os.path.join(route, 'rgb_right')))
        os.system('rm -rf %s' % (os.path.join(route, 'rgb_rear')))
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
