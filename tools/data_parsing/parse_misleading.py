import os
import re
import sys
import json
from multiprocessing import Pool

from tqdm import tqdm

from turn_rules import Turn01, Turn02, Turn03, Turn04, Turn05, TurnFalse, IslandFalse
from follow_rules import Follow01, Follow02, Follow03, Follow04, ChangeLaneFalse, HighwayFalse, EnterHighwayFalse, AccelerateFalse
from notice_rules import Notice01, Notice02, Notice03, Notice04, Notice05, Notice06, Notice07, Notice08
from other_rules import Other01, Other02, Other03, Other04, Other05

registered_class = {
    'Turn-04-L': TurnFalse(direction='left'),
    'Turn-04-R': TurnFalse(direction='right'),

    'Follow-01-L': ChangeLaneFalse(direction='left'),
    'Follow-01-R': ChangeLaneFalse(direction='right'),
    'Follow-02-s2': HighwayFalse(),
    'Follow-04-L': EnterHighwayFalse(),
    'Follow-04-R': EnterHighwayFalse(),

    'Turn-01-L': TurnFalse(direction='left'),
    'Turn-01-R': TurnFalse(direction='right'),

    'Turn-05-1': IslandFalse(),
    'Turn-05-2': IslandFalse(),
    'Turn-05-3': IslandFalse(),

    'Other-06': AccelerateFalse()
}

rule_id_mapping_dict = {}
for i, key in enumerate(registered_class.keys()):
    rule_id_mapping_dict[key] = i

processing_rules = ['Turn-01-L','Turn-01-R','Turn-04-L','Turn-04-R','Follow-01-L','Follow-01-R','Turn-05-1','Turn-05-2','Turn-05-3','Follow-02-s2','Follow-04-L','Follow-04-R','Other-06']

def process(line):
    try:
        processed_data = []
        path, frames = line.split()
        dir_path = os.path.join(dataset_root, path)
        frames = int(frames.strip())
        town_id = int(re.findall(r'town(\d\d)', dir_path)[0])
        weather_id = int(re.findall(r'_w(\d+)_', dir_path)[0])
        json_data = []
        for frame_id in tqdm(range(frames)):
            full_path = os.path.join(dir_path, "measurements_full","%04d.json" % frame_id)
            value_json = json.load(open(full_path, 'r'))
            json_data.append(value_json)

        for rule in processing_rules:
            results = registered_class[rule].process({'data': json_data, 'town_id': town_id, 'weather_id': weather_id})
            rule_id = rule_id_mapping_dict[rule]
            for result in results:
                result['instruction'] = rule
                result['instruction_id'] = rule_id
                result['town_id'] = town_id
                result['weather_id'] = weather_id
                result['route_path'] = path
                result['route_frames'] = frames
                result['bad_case'] = 'True'
                processed_data.append(result)

        return processed_data

    except Exception as e:
        except_type, except_value, except_traceback = sys.exc_info()
        except_file = os.path.split(except_traceback.tb_frame.f_code.co_filename)[1]
        exc_dict = {
            "Error type: ": except_type,
            "Error information": except_value,
            "Error file": except_file,
            "Error line": except_traceback.tb_lineno,
        }
        print(exc_dict)
        print(town_id)
        print(rule)
        print(full_path)

if __name__ == '__main__':
    dataset_root = sys.argv[1]
    list_file = os.path.join(dataset_root, 'dataset_index.txt')
    lines = open(list_file, 'r').readlines()

    with Pool(8) as p:
        r_list = list(tqdm(p.imap(process, lines), total=len(lines)))

    f_write = open(os.path.join(dataset_root, 'misleading_data.txt'), 'w')
    for results in r_list:
        if not results:
            continue
        for result in results:
            processed_json = json.dumps(result)
            f_write.write(processed_json+'\n')
    f_write.close()
