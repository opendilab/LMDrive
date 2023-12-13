import math
import random
random.seed(0)

class Other01():
    def __init__(self):
        self.total_range = 128

    def add_result(self, results, index, frames):
        frame = frames[index]
        last_frame = frames[index - 1]
        step = math.sqrt(pow(frame["x"]-last_frame["x"], 2)+pow(frame["y"]-last_frame["y"], 2))
        if step == 0:
            return index
        instruction_args = []
        result = {}
        result["start_frame"] = index - 1
        if self.total_range + result["start_frame"] >= len(frames):
            self.total_range = 32 + int((len(frames)-1-(result["start_frame"]+32))*random.random())
        result["end_frame"] = min(result["start_frame"]+self.total_range, len(frames)-1)
        result["instruction_args"] = instruction_args
        results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        index = 1
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
            if results:
                return results
        return results

class Other02():
    def __init__(self):
        self.total_range = float('inf')

    def sample_frame(self, index, frames):
        start = index
        frame_range = min(start+self.total_range,len(frames))
        end = frame_range - 1

        for i in range(start, frame_range):
            frame = frames[i]
            last_frame = frames[i - 1]
            if math.sqrt(pow(frame["x"]-last_frame["x"], 2)+pow(frame["y"]-last_frame["y"], 2)) != 0 and frame["brake"] == True:
                continue
            end = i - 1
            return start, end

        return start, end

    def add_result(self, results, index, frames):
        frame = frames[index]
        last_frame = frames[index - 1]
        step = math.sqrt(pow(frame["x"]-last_frame["x"], 2)+pow(frame["y"]-last_frame["y"], 2))
        if step == 0:
            return index
        if frame["brake"] != True:
            return index
        start_frame, end_frame = self.sample_frame(index, frames)
        index = end_frame
        if end_frame - start_frame < 8:
            return index
        instruction_args = []
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
            if results:
                return results
        return results

class Other03():
    def __init__(self):
        self.total_range = float('inf')

    def sample_frame(self, index, frames):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = None

        for i in range(start, frame_range):
            frame = frames[i]
            last_frame = frames[i - 1]
            if frame["brake"]!=True:
                return start, end
            if math.sqrt(pow(frame["x"]-last_frame["x"], 2)+pow(frame["y"]-last_frame["y"], 2)) != 0:
                continue
            end = i
            return start, end

        return start, end

    def add_result(self, results, index, frames):
        frame = frames[index]
        last_frame = frames[index - 1]
        step = math.sqrt(pow(frame["x"]-last_frame["x"], 2)+pow(frame["y"]-last_frame["y"], 2))
        if step == 0:
            return index
        if frame["brake"] != True:
            return index
        start_frame, end_frame = self.sample_frame(index, frames)
        if end_frame == None:
            return index
        index = end_frame
        if end_frame - start_frame < 8:
            return index
        instruction_args = []
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
            if results:
                return results
        return results

class Other04():
    def __init__(self):
        self.total_range = 128

    def add_result(self, results, index, frames):
        pro = random.random()
        rand = random.random()
        if pro > 0.1:
            start = index
            end = min(int(start+self.total_range*rand), len(frames)-1)
            index = end
            if end - start < 32:
                return index
            instruction_args = []
            result = {}
            result["start_frame"] = start
            result["end_frame"] = end
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

class Other05():
    def __init__(self):
        self.total_range = float('inf')
        self.angle = [0,math.pi/2,math.pi,math.pi*3/2,math.pi*2]
        self.angle_instruction_mapping = [[1,0,0],[0,1,1],[1,0,1],[0,1,0]]
        self.left_right_mapping = [["right","left"],["left","right"]]

    def sample_frame(self, index, frames, des_loc):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = None
        distance_min = float('inf')

        for i in range(start, frame_range):
            frame = frames[i]
            eagle_loc = [frame["gps_y"],frame["gps_x"]]
            distance = math.sqrt(pow(eagle_loc[0]-des_loc[0], 2)+pow(eagle_loc[1]-des_loc[1], 2))
            if distance < distance_min:
                distance_min = distance
                i_min = i

        if distance_min < 2:
            end = i_min

        return start, end

    def add_result(self, results, index, frames):
        frame = frames[index]
        eagle_loc = [frame["gps_y"],frame["gps_x"]]
        des_loc = [frame["y_command"],frame["x_command"]]
        pro = random.random()
        if pro <= 0.1:
            return index
        for i in range(4):
            if abs(frame["theta"]-self.angle[i]) > 0.5 and 2*math.pi - abs(frame["theta"]-self.angle[i]) > 0.5:
                continue
            straight = round(abs(des_loc[self.angle_instruction_mapping[i][0]]-eagle_loc[self.angle_instruction_mapping[i][0]]))
            horizontal = des_loc[self.angle_instruction_mapping[i][1]]-eagle_loc[self.angle_instruction_mapping[i][1]]
            if horizontal > 0:
                direction = self.left_right_mapping[self.angle_instruction_mapping[i][2]][0]
            else:
                direction = self.left_right_mapping[self.angle_instruction_mapping[i][2]][1]
            horizontal = round(abs(horizontal))
            start_frame, end_frame = self.sample_frame(index, frames, des_loc)
            if end_frame == None:
                return index
            if end_frame - start_frame < 4 or end_frame - start_frame > 128:
                return index
            angle_start = frames[start_frame]["theta"]
            angle_end = frames[end_frame]["theta"]
            if abs(angle_end-angle_start) < 2*math.pi/72 or 2*math.pi - abs(angle_end-angle_start) < 2*math.pi/72:
                direction = "straight"
            instruction_args = [straight,direction,horizontal]
            index = end_frame
            result = {}
            result["start_frame"] = start_frame
            result["end_frame"] = end_frame
            result["instruction_args"] = instruction_args
            results.append(result)
            return index

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

