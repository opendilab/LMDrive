import os
import sys

from tqdm import tqdm

if __name__ == "__main__":
    dataset_root = sys.argv[1]
    list_file = os.path.join(dataset_root, 'dataset_index.txt')
    routes = os.listdir(dataset_root)
    with open(list_file, 'w') as f:
        for route in tqdm(routes):
            if os.path.isdir(os.path.join(dataset_root, route)):
                frames = len(os.listdir(os.path.join(dataset_root, route, 'measurements')))
                if frames < 32:
                    print("Route %s only havs %d frames (<32). We have omitted it!" % (route, frames))
                else:
                    f.write(route + ' ' + str(frames) + '\n')
