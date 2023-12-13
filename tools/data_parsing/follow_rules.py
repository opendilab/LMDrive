import os
import math
import carla
import random
from pathlib import Path
random.seed(0)
from abc import ABC, abstractmethod

class Follow(ABC):
    @abstractmethod
    def choose_start(self, index, frames, distance):
        """ Choose start point of data clip that has instruction with distance"""
        start = None
        initial_frame = frames[index]
        difference_min = float('inf')

        for i in range(index):
            frame = frames[index - i]
            dist = math.sqrt(pow(initial_frame["x"]-frame["x"], 2)+pow(initial_frame["y"]-frame["y"], 2))
            if abs(dist-distance) < difference_min:
                difference_min = abs(dist-distance)
                index_min = i
            elif difference_min < 2:
                break

        if difference_min < 2:
            start = index - index_min

        return start

class Follow01(Follow):
    def __init__(self, direction, dis=False):
        self.direction = direction
        self.dis = dis
        self.direction_command_mapping = {"left": 5, "right": 6}
        self.total_range = float('inf')

    def choose_start(self, index, frames, distance):
        start = super().choose_start(index, frames, distance)
        return start

    def sample_frame(self, index, frames):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = frame_range - 1
        distance = 2 + 18*random.random()
        instruction_args = []

        for i in range(start, frame_range):
            frame = frames[i]
            command = frame["command"]
            # Endpoint of the sampling
            if command == self.direction_command_mapping[self.direction]:
                continue
            if command != 3 and command != 4:
                end = None
                return start, end, instruction_args
            if abs(frames[start]["theta"]-frame["theta"]) > 1/36*math.pi and 2*math.pi - abs(frames[start]["theta"]-frame["theta"]) > 1/36*math.pi:
                continue
            end = i
            if self.dis == False:
                return start, end, instruction_args
            start = self.choose_start(index, frames, distance)
            instruction_args.append(round(distance))
            return start, end, instruction_args

        if self.dis == False:
            return start, end, instruction_args
        start = self.choose_start(index, frames, distance)
        instruction_args.append(round(distance))
        return start, end, instruction_args

    def add_result(self, results, index, frames):
        frame = frames[index]
        command = frame["command"]
        if command != self.direction_command_mapping[self.direction]:
            return index
        if frames[index - 1]["command"] != 3 and frames[index - 1]["command"] != 4:
            return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
        if end_frame == None or start_frame == None:
            return index
        index = end_frame  # Move forward the index
        period = end_frame - start_frame
        if period < 32:
            end_frame = min(end_frame+32-period, len(frames)-1)
            period = end_frame - start_frame
        if period < 16:
            return index
        result = {}
        result["start_frame"] = start_frame
        result["end_frame"] = end_frame
        result["instruction_args"] = instruction_args
        results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
        return results

class Follow02(Follow):
    def __init__(self, style, dis=False):
        self.style = style
        self.dis = dis
        self.total_range = float('inf')
        self.highway_range = {"4": [[-487.84,361.47,2.84,44.26],[-19.73,18.43,-279.10,278.82],[94.88,333.41,-360.95,-398.73],[-376.92,-93.35,400.16,440.26],[-517.71,-478.04,37.87,319.51]],\
                              "5": [[-257.43,-217.99,-179.75,175.86],[184.14,218.66,-175.28,174.62],[-204.68,162.79,-217.05,-181.86],[-210.39,179.10,182.47,218.67]],\
                              "6": [[-302.75,625.72,-8.10,-26.18],[-278.19,651.95,35.46,54.74],[-286.53,649.00,135.70,155.15],[-323.99,647.78,236.17,254.11],[656.33,673.07,12.02,228.79],[-372.63,-359.37,13.01,230.13]]}

    def check_highway(self, frame, town_id):
        for range in self.highway_range[str(town_id)]:
            if frame["x"] > range[0] and frame["x"] < range[1] and frame["y"] > range[2] and frame["y"] < range[3]:
                return True

        return False

    def choose_start(self, index, frames, distance):
        pass

    def sample_frame(self, index, frames):
        start = index
        frame_range = min(start + self.total_range, len(frames))
        end = frame_range - 1
        distance = 2 + 18*random.random()
        pass_distance = 0
        instruction_args = []

        for i in range(start, frame_range):
            frame = frames[i]
            if self.dis == True:
                last_frame = frames[i - 1]
                step = math.sqrt(pow(frame["x"]-last_frame["x"], 2)+pow(frame["y"]-last_frame["y"], 2))
                pass_distance = pass_distance + step
                if pass_distance < distance:
                    continue
                end = i
                instruction_args.append(round(distance))
                return start, end, instruction_args
            command = frame["command"]
            # Endpoint of the sampling
            if command == 4 or command == 3:
                continue
            end = i
            return start, end, instruction_args

        if self.dis == False:
            return start, end, instruction_args
        distance = pass_distance
        instruction_args.append(round(distance))
        return start, end, instruction_args

    def add_result(self, results, index, frames, town_id):
        frame = frames[index]
        command = frame["command"]
        if self.style == 2 and not self.check_highway(frame, town_id) :
            return index
        if command != 4:
            return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
        if end_frame == None or start_frame == None:
            return index
        index = end_frame  # Move forward the index
        if end_frame - start_frame < 32:
            return index
        result = {}
        result["start_frame"] = start_frame
        result["end_frame"] = end_frame
        result["instruction_args"] = instruction_args
        results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        town_id = data["town_id"]
        if self.style == 2:
            if town_id < 4 or town_id > 6:
                return results
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames, town_id)
            index = index + 1
        return results

class Follow03(Follow):
    def __init__(self, style, dis=False):
        self.style = style
        self.dis = dis
        self.total_range = float('inf')

    def choose_start(self, index, frames, distance):
        pass

    def sample_frame(self, index, frames):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = frame_range - 1
        distance = 4 + 36*random.random()
        pass_distance = 0
        instruction_args = []

        for i in range(start, frame_range):
            frame = frames[i]
            command = frame["command"]
            if self.dis == True:
                last_frame = frames[i - 1]
                step = math.sqrt(pow(frame["x"]-last_frame["x"], 2)+pow(frame["y"]-last_frame["y"], 2))
                pass_distance = pass_distance + step
                if pass_distance >= distance:
                    instruction_args.append(round(pass_distance))
                    return start, end, instruction_args
                if command == 4 :
                    continue
                if command == 3 and self.style == 1:
                    continue
                end = i
                instruction_args.append(round(pass_distance))
                return start, end, instruction_args

            if command == 4 :
                continue
            if command == 3 and self.style == 1:
                continue
            end = i
            return start, end, instruction_args

        if self.dis == False:
            return start, end, instruction_args
        distance = pass_distance
        instruction_args.append(round(distance))
        return start, end, instruction_args

    def add_result(self, results, index, frames):
        frame = frames[index]
        command = frame["command"]
        if self.style == 2 and command != 4:
            return index
        if command != 3 and command != 4:
            return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
        if end_frame == None or start_frame == None:
            return index
        index = end_frame  # Move forward the index
        if end_frame - start_frame < 32:
            return index
        result = {}
        result["start_frame"] = start_frame
        result["end_frame"] = end_frame
        result["instruction_args"] = instruction_args
        results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
        return results

class Follow04(Follow):
    def __init__(self, directions, dis=False):
        self.directions = directions
        self.dis = dis
        self.total_range = float('inf')
        self.townid_loc_mapping = {"right": {"1":None,"2":None,"3":None,"4":[{"in":[15.56,-56.04],"out":[75.02,6.16]},{"in":[90.57,39.99],"out":[16.75,99.37]},{"in":[-16.52,105.38],"out":[-76.96,37.67]},{"in":[-83.45,5.42],"out":[-15.78,-50.79]},{"in":[-6.80,-277.01],"out":[-383.98,1.90]},{"in":[-7.44,327.68],"out":[404.62,6.40]}],"5":None,"6":None,"7":None,"10":None},\
                                   "left": {"1":None,"2":None,"3":None,"4":None,"5":None,"6":[{"in":[-137.44,-8.89],"out":[-140.42,42.52]},{"in":[132.52,38.19],"out":[134.92,-15.95]},{"in":[494.16,37.50],"out":[506.05,-12.70]},{"in":[-211.54,149.29],"out":[-211.91,236.81]},{"in":[-111.28,237.35],"out":[-111.96,148.93]}],"7":None,"10":None},\
                                   "straight": {"1":None,"2":None,"3":None,"4":None,"5":None,"6":[{"in":[81.21,135.94],"out":[257.00,52.54]},{"in":[469.45,137.07],"out":[549.61,52.18]},{"in":[98.99,236.78],"out":[243.89,151.35]}],"7":None,"10":None}}

        self.direction_command_mapping = {"left":1, "right":2, "straight":3}

    def choose_start(self, index, frames, distance):
        start = super().choose_start(index, frames, distance)
        return start

    def sample_frame(self, index, frames, des_loc):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = frame_range - 1
        distance = 2 + 18*random.random()
        instruction_args = []

        for i in range(start, frame_range):
            frame = frames[i]
            if math.sqrt(pow(des_loc["out"][0]-frame["x"], 2)+pow(des_loc["out"][1]-frame["y"], 2))>10:
                continue
            end = i
            if self.dis == False:
                return start, end, instruction_args
            start = self.choose_start(index, frames, distance)
            instruction_args.append(round(distance))
            return start, end, instruction_args

        if self.dis == False:
            return start, end, instruction_args
        start = self.choose_start(index, frames, distance)
        instruction_args.append(round(distance))
        return start, end, instruction_args

    def add_result(self, results, index, frames, town_id, direction):
        frame = frames[index]
        command = frame["command"]
        town_loc = self.townid_loc_mapping[direction][str(town_id)]

        for des_loc in town_loc:
            if math.sqrt(pow(des_loc["in"][0]-frame["x"], 2)+pow(des_loc["in"][1]-frame["y"], 2))>10:
                continue
            if command != self.direction_command_mapping[direction]:
                return index
            start_frame, end_frame, instruction_args = self.sample_frame(index, frames, des_loc)
            if end_frame == None or start_frame == None:
                return index
            index = end_frame  # Move forward the index
            if end_frame - start_frame < 32:
                return index
            result = {}
            result["start_frame"] = start_frame
            result["end_frame"] = end_frame
            result["instruction_args"] = instruction_args
            results.append(result)

        return index

    def process(self, data):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        town_id = data["town_id"]
        for direction in self.directions:
            if self.townid_loc_mapping[direction][str(town_id)] == None:
                continue
            index = 15
            while index < frame_num :
                index = self.add_result(results, index, frames, town_id, direction)
                index = index + 1
        return results

class ChangeLaneFalse():
    def __init__(self, direction):
        self.direction = direction
        self.total_range = 128
        self.max_range = 512

    def choose_start(self, index, frames, distance):
        start = None
        if index == None:
            return start
        initial_frame = frames[index]
        difference_min = float('inf')

        for i in range(min(index,self.max_range)):
            frame = frames[index - i]
            dist = math.sqrt(pow(initial_frame["x"]-frame["x"], 2)+pow(initial_frame["y"]-frame["y"], 2))
            if abs(dist-distance) < difference_min:
                difference_min = abs(dist-distance)
                index_min = i
            elif difference_min < 2:
                break

        if difference_min < 2:
            start = index - index_min

        return start

    def turn_end(self, frame):
        if abs(0-frame["theta"]) > 1/36 * math.pi and 2*math.pi - abs(0-frame["theta"]) > 1/36 * math.pi and abs(math.pi*1/2-frame["theta"]) > 1/36 * math.pi and 2*math.pi - abs(math.pi*1/2-frame["theta"]) > 1/36 * math.pi and abs(math.pi-frame["theta"]) > 1/36 * math.pi and 2 * math.pi - abs(math.pi-frame["theta"]) > 1/36 * math.pi and abs(math.pi*3/2-frame["theta"]) > 1/36 * math.pi and 2*math.pi - abs(math.pi*3/2-frame["theta"]) > 1/36 * math.pi:
            return True

    def get_left_nums(self, waypoint):
        num = 0
        last_lane_id = waypoint.lane_id
        while True:
            waypoint = waypoint.get_left_lane()
            if waypoint == None:
                break
            if last_lane_id*waypoint.lane_id < 0:
                break
            last_lane_id = waypoint.lane_id
            if str(waypoint.lane_type) != "Driving" :
                break
            num += 1
        return num

    def get_right_nums(self, waypoint):
        num = 0
        while True:
            waypoint = waypoint.get_right_lane()
            if waypoint == None:
                break
            if str(waypoint.lane_type) != "Driving" :
                break
            num += 1
        return num

    def sample_turn_frame(self, index, frames):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = frame_range - 1
        distance = 2 + 18 * random.random()
        instruction_args = []

        for i in range(start, frame_range):
            frame = frames[i]
            command = frame["command"]
            if command in [1,2,3]:  # Endpoint of the sampling
                continue
            if self.turn_end(frame):
                continue
            end = i
            if end - start > self.max_range:
                start = self.skip_red_light(frames, start, end)
            return start, end, instruction_args
        if end - start > self.max_range:
            start = self.skip_red_light(frames, start, end)
        return start, end, instruction_args

    def sample_frame(self, index, frames):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = frame_range - 1
        distance = 2 + 18*random.random()
        pass_distance = 0
        instruction_args = []

        for i in range(start, frame_range):
            frame = frames[i]
            command = frame["command"]
            # Endpoint of the sampling
            if command == 4 or command == 3:
                continue
            end = i
            return start, end, instruction_args

        return start, end, instruction_args

    def sample_frame_multi(self, index, frames, map_file):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = frame_range - 1
        distance = 2 + 18*random.random()
        pass_distance = 0
        instruction_args = []

        for i in range(start, frame_range):
            frame = frames[i]
            location = carla.Location(x=frame["x"], y=frame["y"], z=0.00)
            waypoint = map_file.get_waypoint(location)
            if self.direction == 'left':
                if self.get_left_nums(waypoint) != 0:
                    end = i
                    return start, end, instruction_args
            elif self.direction == 'right':
                if self.get_right_nums(waypoint) != 0:
                    end = i
                    return start, end, instruction_args
            command = frame["command"]
            # Endpoint of the sampling
            if command == 4 or command == 3:
                continue
            end = i
            return start, end, instruction_args
        return start, end, instruction_args

    def add_result(self, results, index, frames):
        frame = frames[index]
        command = frame["command"]
        if command in [1,2,3]:
            start_frame, end_frame, instruction_args = self.sample_turn_frame(index, frames)
            if end_frame == None:
                return index
            if start_frame == None:
                index = end_frame
                return index
            index = end_frame  # Move forward the index
            if end_frame - start_frame < 32:
                return index
            if random.random() > 0.96:
                result = {}
                result["start_frame"] = start_frame
                result["end_frame"] = end_frame
                result["instruction_args"] = instruction_args
                results.append(result)
            return index
        if command != 4:
            return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
        if end_frame == None or start_frame == None:
            return index
        index = end_frame  # Move forward the index
        if end_frame - start_frame < 32:
            return index
        if random.random() > 0.975:
            result = {}
            result["start_frame"] = start_frame
            result["end_frame"] = end_frame
            result["instruction_args"] = instruction_args
            results.append(result)
        return index

    def add_result_multi(self, results, index, frames, map_file):
        frame = frames[index]
        location = carla.Location(x=frame["x"], y=frame["y"], z=0.00)
        waypoint = map_file.get_waypoint(location)
        command = frame["command"]
        if command in [1,2,3]:
            start_frame, end_frame, instruction_args = self.sample_turn_frame(index, frames)
            if end_frame == None:
                return index
            if start_frame == None:
                index = end_frame
                return index
            index = end_frame  # Move forward the index
            if end_frame - start_frame < 32:
                return index
            if random.random() > 0.96:
                result = {}
                result["start_frame"] = start_frame
                result["end_frame"] = end_frame
                result["instruction_args"] = instruction_args
                results.append(result)
            return index
        if command != 4:
            return index
        if self.direction == 'left':
            if self.get_left_nums(waypoint) != 0:
                return index
        elif self.direction == 'right':
            if self.get_right_nums(waypoint) != 0:
                return index
        start_frame, end_frame, instruction_args = self.sample_frame_multi(index, frames, map_file)
        if end_frame == None or start_frame == None:
            return index
        index = end_frame  # Move forward the index
        if end_frame - start_frame < 32:
            return index
        if random.random() > 0.975:
            result = {}
            result["start_frame"] = start_frame
            result["end_frame"] = end_frame
            result["instruction_args"] = instruction_args
            results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        town_id = data["town_id"]
        if town_id not in [1,2,7]:
            curr_dir = Path(__file__).parent
            map_file_name = "%02d.xodr" % (town_id)
            map_file_dir = os.path.join(curr_dir, "carla_maps", map_file_name)
            with open(map_file_dir, 'r') as fp:
                map_file = carla.Map('Town%02d' % (town_id), fp.read())
            index = 15
            while index < frame_num :
                index = self.add_result_multi(results, index, frames, map_file)
                index = index + 1
            return results
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
        return results

class HighwayFalse():
    def __init__(self):
        super().__init__()
        self.total_range = 128

    def add_result(self, results, index, frames):
        rand = random.random()
        start = index
        end = min(int(start+32+self.total_range*rand), len(frames)-1)
        index = end
        if end - start < 32:
            return index
        instruction_args = []
        if random.random() > 0.99:
            result = {}
            result["start_frame"] = start
            result["end_frame"] = end
            result["instruction_args"] = instruction_args
            results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        town_id = data["town_id"]
        if town_id in ["4","5","6"]:
            return results
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
        return results

class EnterHighwayFalse():
    def __init__(self):
        super().__init__()
        self.total_range = 128

    def add_result(self, results, index, frames):
        rand = random.random()
        start = index
        end = min(int(start+32+self.total_range*rand), len(frames)-1)
        index = end
        if end - start < 32:
            return index
        instruction_args = []
        if random.random() > 0.99:
            result = {}
            result["start_frame"] = start
            result["end_frame"] = end
            result["instruction_args"] = instruction_args
            results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        town_id = data["town_id"]
        if town_id in ["4","5","6"]:
            return results
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
        return results

class AccelerateFalse():
    def __init__(self):
        self.total_range = 128

    def sample_frame(self, index, frames):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = frame_range - 1
        instruction_args = []
        for i in range(start, frame_range):
            frame = frames[i]
            # Endpoint of the sampling
            if frame["brake"] == True:
                continue
            end = i - 1
            return start, end, instruction_args
        return start, end, instruction_args

    def add_result(self, results, index, frames):
        frame = frames[index]
        if frame["brake"] == False:
            return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
        if end_frame == None or start_frame == None:
            return index
        index = end_frame  # Move forward the index
        period = end_frame - start_frame
        if period < 32:
            return index
        if random.random() > 0.97:
            result = {}
            result["start_frame"] = start_frame
            result["end_frame"] = end_frame
            result["instruction_args"] = instruction_args
            results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
        return results
