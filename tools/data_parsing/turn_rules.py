import time
import math
import random
random.seed(0)
from abc import ABC, abstractmethod

class Turn(ABC):
    def __init__(self):
        self.max_range = 512  # Maximum frame length of turning instruction data clip

    @abstractmethod
    def choose_start(self, index, frames, distance):
        """ Choose start point of data clip that has instruction with distance"""
        start = None
        if index == None:
            return start
        initial_frame = frames[index]
        difference_min = float('inf')

        for i in range(min(index, self.max_range)):
            frame = frames[index - i]
            dist = math.sqrt(pow(initial_frame["x"]-frame["x"], 2) + pow(initial_frame["y"]-frame["y"], 2))
            if abs(dist - distance) < difference_min:
                difference_min = abs(dist-distance)
                index_min = i
            elif difference_min < 2:
                break
        
        if difference_min < 2:
            start = index - index_min

        return start     

    @abstractmethod
    def turn_end(self, frame):
        """ Check if vehicle terminated the turning instruction"""
        if abs(0-frame["theta"]) > 1/36 * math.pi and 2*math.pi - abs(0-frame["theta"]) > 1/36 * math.pi and abs(math.pi*1/2-frame["theta"]) > 1/36 * math.pi and 2*math.pi - abs(math.pi*1/2-frame["theta"]) > 1/36 * math.pi and abs(math.pi-frame["theta"]) > 1/36 * math.pi and 2*math.pi - abs(math.pi-frame["theta"]) > 1/36 * math.pi and abs(math.pi*3/2-frame["theta"]) > 1/36*math.pi and 2*math.pi - abs(math.pi*3/2-frame["theta"]) > 1/36 * math.pi:
            return True
        
    @abstractmethod
    def skip_red_light(self, frames, start, end):
        """ If vehicle stopped for a long time waiting at a red light, discard that portion of data"""
        distance = math.sqrt(pow(frames[start]["x"]-frames[end - self.max_range]["x"], 2)+pow(frames[start]["y"]-frames[end-self.max_range]["y"], 2))
        if distance <= 4:
            start = end - self.max_range
        else:
            start = None
        return start

class Turn01(Turn):
    def __init__(self, direction, dis=False):
        super().__init__()
        self.direction = direction
        self.dis = dis  # Generate instruction with or without distance
        self.direction_command_mapping = {"left": 1, "right": 2}
        self.total_range = float('inf')

    def choose_start(self, index, frames, distance):
        start = super().choose_start(index, frames, distance)
        return start
    
    def turn_end(self, frame):
        return super().turn_end(frame)        

    def skip_red_light(self, frames, start, end):
        start = super().skip_red_light(frames, start, end)
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
            if command == self.direction_command_mapping[self.direction]:  # Endpoint of the sampling
                continue
            if self.turn_end(frame):
                continue
            end = i
            if end - start > self.max_range:
                start = self.skip_red_light(frames, start, end)
            if self.dis == False:
                return start, end, instruction_args
            start = self.choose_start(start, frames, distance)
            instruction_args.append(round(distance))
            return start, end, instruction_args

        if end - start > self.max_range:
            start = self.skip_red_light(frames, start, end)        
        if self.dis == False:
            return start, end, instruction_args
        start = self.choose_start(start, frames, distance)
        instruction_args.append(round(distance))
        return start, end, instruction_args
    
    def add_result(self, results, index, frames):
        frame = frames[index]
        command = frame["command"]
        if command != self.direction_command_mapping[self.direction]:
            return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
        if end_frame == None:
            return index
        if start_frame == None:
            index = end_frame
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

class Turn02(Turn):
    def __init__(self, direction, dis=False):
        super().__init__()    
        self.direction = direction
        self.dis = dis
        self.direction_command_mapping = {"left": 1, "right": 2, "straight": 3}
        self.total_range = float('inf')
        self.townid_loc_mapping = {"1": None,"2": None,"3": [[-226.23,-2.30],[-223.15,103.26],[83.79,-257.12],[157.84,-256.18],[-146.60,-1.44],[-84.86,133.58],[-2.82,132.36],[-81.72,-137.82],[2.44,-135.59],[83.89,-135.75],[85.39,-199.39],[153.65,-198.61]],"4": [[203.01,-309.33],[202.12,-247.58],[200.61,-171.29],[256.94,-248.01],[256.49,-170.93],[313.26,-248.37]],"5": [[-189.88,-90.40],[-189.49,0.79],[-190.41,89.65],[-127.13,-89.45],[-126.58,1.19],[-125.56,89.59],[-49.85,-89.76],[-49.13,0.86],[-49.28,89.65],[31.55,-89.33],[29.53,0.28],[29.20,89.69],[101.55,-0.07]],"6": [[-1.63,-17.53],[-1.84,49.77]],"7": [[-101.62,53.08],[-3.78,-1.48],[-150.54,-35.13]],"10": [[-47.38,19.22]]}

    def choose_start(self, index, frames, distance):
        start = super().choose_start(index, frames, distance)
        return start

    def turn_end(self, frame):
        return super().turn_end(frame)

    def skip_red_light(self, frames, start, end):
        start = super().skip_red_light(frames, start, end)
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
            if command == self.direction_command_mapping[self.direction]:  # Endpoint of the sampling
                continue
            if self.turn_end(frame):
                continue
            end = i
            if end - start > self.max_range:
                start = self.skip_red_light(frames, start, end)
            if self.dis == False:
                return start, end, instruction_args
            start = self.choose_start(start, frames, distance)
            instruction_args.append(round(distance))
            return start, end, instruction_args

        if end - start > self.max_range:
            start = self.skip_red_light(frames, start, end)        
        if self.dis == False:
            return start, end, instruction_args
        start = self.choose_start(start, frames, distance)
        instruction_args.append(round(distance))
        return start, end, instruction_args

    def add_result(self, results, index, frames, town_id):
        frame = frames[index]
        command = frame["command"]
        town_loc = self.townid_loc_mapping[str(town_id)]
        if command != self.direction_command_mapping[self.direction]:
            return index
        
        for des_loc in town_loc:
            if math.sqrt(pow(des_loc[0]-frame["x"], 2)+pow(des_loc[1]-frame["y"], 2))>35:
                continue
            start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
            if end_frame == None:
                return index
            if start_frame == None:
                index = end_frame
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
        
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        town_id = data["town_id"]
        if self.townid_loc_mapping[str(town_id)] == None:
            return results
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames, town_id)
            index = index + 1
        return results

class Turn03(Turn):
    def __init__(self, direction, dis=False):
        super().__init__()
        self.direction = direction
        self.dis = dis
        self.direction_command_mapping = {"left": 1, "right": 2, "straight": 3}
        self.total_range = float('inf')

    def choose_start(self, index, frames, distance):
        start = super().choose_start(index, frames, distance)
        return start

    def turn_end(self, frame):
        return super().turn_end(frame)

    def skip_red_light(self, frames, start, end):
        start = super().skip_red_light(frames, start, end)
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
            if command == self.direction_command_mapping[self.direction]:  # Endpoint of the sampling
                continue
            if self.turn_end(frame):
                continue
            end = i
            if end - start > self.max_range:
                start = self.skip_red_light(frames, start, end)
            if self.dis == False:
                return start, end, instruction_args
            start = self.choose_start(start, frames, distance)
            instruction_args.append(round(distance))
            return start, end, instruction_args

        if end - start > self.max_range:
            start = self.skip_red_light(frames, start, end)
        if self.dis == False:
            return start, end, instruction_args        
        start = self.choose_start(start, frames, distance)
        instruction_args.append(round(distance))
        return start, end, instruction_args

    def add_result(self, results, index, frames):
        frame = frames[index]
        command = frame["command"]        
        if command != self.direction_command_mapping[self.direction]:
            return index
        affected_light_id = frame["affected_light_id"]
        if affected_light_id == -1:
            return index
        if not str(affected_light_id) in frame["actors_data"].keys():
            return index
        light_loc = frame["actors_data"][str(affected_light_id)]["loc"]
        light_dis = round(math.sqrt(pow(light_loc[0]-frame["x"], 2)+pow(light_loc[1]-frame["y"], 2)))
        if light_dis > 40:
            return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
        if end_frame == None:
            return index
        if start_frame == None:
            index = end_frame
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
            index = index +1
        return results

class Turn04(Turn):
    def __init__(self, direction, dis=False):
        super().__init__()
        self.direction = direction
        self.dis = dis
        self.direction_command_mapping = {"left": 1, "right": 2, "straight": 3}
        self.total_range = float('inf')
        # element of loc: [x,y,range]
        self.townid_loc_mapping = {"1": [[90.30,0.51,25],[156.93,1.09,25],[336.86,1.39,25],[337.33,326.93,40],[90.95,327.01,40],[92.37,196.73,30],[91.87,131.36,30],[92.17,57.97,25],[156.05,55.61,25],[335.12,57.68,25],[335.78,130.58,30],[336.32,196.97,30]],\
        "2": [[43.31,304.10,30],[-5.34,190.45,30],[192.52,189.99,30],[190.68,239.30,30],[134.06,238.50,30],[43.77,238.49,30],[43.51,190.68,30],[133.09,189.44,30]],\
        "3": [[3.93,-199.79,35],[236.43,0.77,30],[237.27,61.02,30],[-1.39,196.76,35],[151.53,-132.98,30],[149.59,-72.75,30],[80.90,-74.44,30],[148.68,-5.98,30],[78.58,-5.19,30],[169.12,64.11,30]],\
        "4": [[257.15,-308.29,25],[256.30,-122.12,25],[128.78,-172.50,30],[61.36,-174.60,25],[15.01,-172.33,25],[205.67,-364.69,30],[393.50,-171.28,25],[381.09,-67.54,30]],\
        "5": [[34.01,-182.82,20],[40.02,-147.67,20],[153.47,-0.52,25],[40.85,142.48,25],[30.24,198.96,30],[-126.12,-137.57,20],[-124.06,148.97,25],[-268.82,-1.19,30]],\
        "6": [[662.70,41.96,40],[662.41,144.54,40]],\
        "7": [[-197.22,-161.53,40],[-1.85,-238.09,40],[67.08,-1.04,35],[67.25,60.09,35],[-109.01,113.97,35],[-198.61,49.24,25],[-198.65,-36.34,25],[-151.27,48.35,25],[-100.17,-34.76,15],[-100.46,-63.77,15],[-101.47,-96.25,10],[-85.31,-111.70,10],[-73.35,-159.14,30],[-3.43,-159.27,30],[-4.05,-107.83,15],[-4.45,-64.86,20],[-4.79,57.83,35]],\
        "10": [[-44.76,-55.94,30],[96.00,-21.14,30],[96.84,68.01,30],[-46.40,127.21,30],[-99.79,19.70,30],[-38.44,65.96,30],[41.59,66.94,20],[41.08,30.14,20]]}

    def choose_start(self, index, frames, distance):
        start = super().choose_start(index, frames, distance)
        return start

    def turn_end(self, frame):
        return super().turn_end(frame)

    def skip_red_light(self, frames, start, end):
        start = super().skip_red_light(frames, start, end)
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
            if command == self.direction_command_mapping[self.direction]:  # Endpoint of the sampling
                continue
            if self.turn_end(frame):
                continue
            end = i
            if end - start > self.max_range:
                start = self.skip_red_light(frames, start, end)
            if self.dis == False:
                return start, end, instruction_args
            start = self.choose_start(start, frames, distance)
            instruction_args.append(round(distance))
            return start, end, instruction_args

        if end - start > self.max_range:
            start = self.skip_red_light(frames, start, end)
        if self.dis == False:
            return start, end, instruction_args        
        start = self.choose_start(start, frames,distance)
        instruction_args.append(round(distance))
        return start, end, instruction_args

    def add_result(self, results, index, frames, town_id):
        frame = frames[index]
        command = frame["command"]
        town_loc = self.townid_loc_mapping[str(town_id)]
        
        for des_loc in town_loc:
            if math.sqrt(pow(des_loc[0]-frame["x"], 2)+pow(des_loc[1]-frame["y"], 2)) > des_loc[2]:
                continue
            if command != self.direction_command_mapping[self.direction]:
                return index
            start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
            if end_frame == None:
                return index
            if start_frame == None:
                index = end_frame
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
        
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        town_id = data["town_id"]
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames, town_id)
            index = index + 1
        return results

class Turn05(Turn):
    def __init__(self, exit_no):
        super().__init__()
        self.exit_no = exit_no
        self.total_range = float('inf')
        self.island_in = [[-34.53,3.25],[5.53,36.44],[37.73,-8.18],[-6.62,-40.55]]
        self.island_out = [[-30.88,-6.80],[-10.26,36.47],[37.36,7.85],[8.98,-37.09]]

    def choose_start(self, index, frames, distance):
        pass

    def turn_end(self, frame):
        return super().turn_end(frame)

    def skip_red_light(self, frames, start, end):
        pass

    def sample_frame(self, index, frames, no_in):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = None
        instruction_args = []
        for i in range(start, frame_range):
            frame = frames[i]
            for no_out in range(4):
                if math.sqrt(pow(self.island_out[no_out][0]-frame["x"], 2)+pow(self.island_out[no_out][1]-frame["y"], 2))>5:
                    continue
                no = no_out - no_in
                if no < 0 : 
                    no = 4 + no
                if no == self.exit_no:
                    end = i
                    return start, end, instruction_args
                return start, end, instruction_args

        return start, end, instruction_args
    
    def add_result(self, results, index, frames):
        frame = frames[index]
        
        for no_in in range(4):
            if math.sqrt(pow(self.island_in[no_in][0]-frame["x"], 2)+pow(self.island_in[no_in][1]-frame["y"], 2))>5:
                continue
            start_frame, end_frame, instruction_args = self.sample_frame(index, frames, no_in)
            if end_frame == None:
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
        
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        town_id = data["town_id"]
        if town_id != 3:
            return results
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
        return results

class Turn06(Turn):
    def __init__(self, first_direction, second_direction):
        super().__init__()
        self.first_direction = first_direction
        self.second_direction = second_direction
        self.direction_command_mapping = {"left": 1, "right": 2, "straight": 3}
        self.total_range = float('inf')

    def choose_start(self, index, frames, distance):
        pass

    def turn_end(self, frame):
        return super().turn_end(frame)

    def skip_red_light(self, frames, start, end):
        start = super().skip_red_light(frames, start, end)
        return start

    def choose_end(self, index, frames):
        frame_range = len(frames)
        end = frame_range - 1

        for i in range(index, frame_range):
            frame = frames[i]
            command = frame["command"]
            if command == self.direction_command_mapping[self.second_direction]:
                continue
            if self.turn_end(frame):
                continue
            end = i
            return end
        
        return end

    def sample_frame(self, index, frames):
        start = index
        frame_range = min(start+self.total_range, len(frames))
        end = None
        instruction_args = []
        flag = False
        
        for i in range(start, frame_range):
            frame = frames[i]
            command = frame["command"]
            if command != self.direction_command_mapping[self.first_direction]:  # First turn end
                flag = True
            if flag == False:
                continue
            if command < 1 or command > 3:
                continue
            if command != self.direction_command_mapping[self.second_direction]:
                return start, end, instruction_args
            end = self.choose_end(i, frames)
            return start, end, instruction_args
        
        return start, end, instruction_args

    def add_result(self, results, index, frames):
        frame = frames[index]
        command = frame["command"]        
        if command != self.direction_command_mapping[self.first_direction]:
            return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames)
        if end_frame == None:
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
            index = index +1
        return results

class TurnFalse(Turn):
    def __init__(self, direction):
        super().__init__()
        self.sample_range = 100
        self.direction = direction
        self.direction_index_mapping = {"left": 0, "right": 1}
        self.direction_command_mapping = {"left": 2, "right": 1}
        self.total_range = 128
        self.Tjunction_loc_mapping = {"1": [[[74.31,2.50,2],[140.33,2.33,2],[320.09,2.32,2],[92.69,72.83,2],[173.38,55.50,2],[334.70,41.42,2],[92.29,147.24,2],[334.53,114.74,2],[92.81,215.65,2],[334.66,179.34,2],[107.98,326.93,2],[354.03,326.60,2]],[[107.74,-1.54,2],[173.88,-1.73,2],[352.97,-1.82,2],[88.03,41.60,2],[139.77,59.79,2],[339.00,73.74,2],[88.19,115.30,2],[339.00,148.54,2],[88.08,181.66,2],[338.96,214.46,2],[73.55,330.99,2],[319.83,331.13,2]]],\
                                   "2": [[[-3.23,205.09,2],[29.32,192.01,2],[117.29,192.16,2],[189.37,175.07,2],[45.80,253.94,2],[148.54,237.29,2],[189.51,225.80,2],[58.12,302.90,2]],[[-7.67,176.45,2],[58.12,187.84,2],[150.38,187.84,2],[193.96,204.03,2],[41.60,223.79,2],[119.50,241.48,2],[193.75,253.49,2],[28.17,307.19,2]]],\
                                    "3": [[[-18.78,-195.73,3],[149.05,-151.50,3],[84.34,-58.19,3],[147.97,-88.64,3],[103.13,-4.93,3],[172.18,-3.80,3],[233.99,-22.35,3],[232.47,37.33,3],[146.53,62.99,2],[22.64,195.33,3]],[[26.14,-205.40,3],[153.95,-116.42,2],[152.70,-58.14,2],[79.87,-88.97,2],[58.55,6.42,3],[129.03,7.83,3],[243.73,27.13,3],[190.25,58.98,2],[241.73,86.48,3],[-24.61,204.59,3]]],\
                                    "4": [[[179.01,-369.13,5],[242.51,-307.31,2],[81.06,-173.40,2],[148.32,-172.98,2],[388.35,-193.76,5],[272.62,-122.34,2],[386.55,-90,5]],[[272.72,-310.59,2],[44.36,-169.96,2],[116.76,-169.19,2],[239.88,-118.47,2]]],\
                                    "5": [[[-149.31,-136.65,3],[-101.78,145.94,3],[7.10,-189.40,5],[55.46,191.61,5],[33.34,162.25,3],[151.32,-18.97,2],[36.20,-129.62,3]],[[-103.21,-143.83,3],[-149.73,153.17,3],[30.16,-165.07,3],[26.36,125.75,3],[155.49,17.09,2]]],\
                                    "6": [[],[[665.92,66.59,5],[664.88,168.47,5]]],\
                                    "7": [[[-197.95,-147.58,2],[-85.62,-158.09,2],[-4.47,-171.80,2],[-5.13,-121.57,2],[-98.38,-50.71,2],[-5.83,-76.91,2],[-199.07,-22.44,2],[-99.01,15.57,2],[-199.03,64.57,2],[-138.96,48.95,2],[10.28,58.64,2],[-95.56,116.56,2]],[[-202.47,-174.34,2],[-202.13,-49.13,2],[-203.63,36.88,2],[-122.42,117.61,2],[-0.51,-145.93,2],[-1.19,-50.03,2]]],\
                                    "10": [[[64.47,64.40,3]],[[19.83,71.62,3]]]}
        self.intersection_loc_mapping = {"1": [[90.30,0.51,25],[156.93,1.09,25],[336.86,1.39,25],[337.33,326.93,40],[90.95,327.01,40],[92.37,196.73,30],[91.87,131.36,30],[92.17,57.97,25],[156.05,55.61,25],[335.12,57.68,25],[335.78,130.58,30],[336.32,196.97,30]],\
        "2": [[43.31,304.10,30],[-5.34,190.45,30],[192.52,189.99,30],[190.68,239.30,30],[134.06,238.50,30],[43.77,238.49,30],[43.51,190.68,30],[133.09,189.44,30]],\
        "3": [[3.93,-199.79,35],[236.43,0.77,30],[237.27,61.02,30],[-1.39,196.76,35],[151.53,-132.98,30],[149.59,-72.75,30],[80.90,-74.44,30],[148.68,-5.98,30],[78.58,-5.19,30],[169.12,64.11,30],[-226.23,-2.30],[-223.15,103.26],[83.79,-257.12],[157.84,-256.18],[-146.60,-1.44],[-84.86,133.58],[-2.82,132.36],[-81.72,-137.82],[2.44,-135.59],[83.89,-135.75],[85.39,-199.39],[153.65,-198.61]],\
        "4": [[257.15,-308.29,25],[256.30,-122.12,25],[128.78,-172.50,30],[61.36,-174.60,25],[15.01,-172.33,25],[205.67,-364.69,30],[393.50,-171.28,25],[381.09,-67.54,30],[203.01,-309.33],[202.12,-247.58],[200.61,-171.29],[256.94,-248.01],[256.49,-170.93],[313.26,-248.37]],\
        "5": [[34.01,-182.82,20],[40.02,-147.67,20],[153.47,-0.52,25],[40.85,142.48,25],[30.24,198.96,30],[-126.12,-137.57,20],[-124.06,148.97,25],[-268.82,-1.19,30],[-189.88,-90.40,30],[-189.49,0.79,30],[-190.41,89.65,30],[-127.13,-89.45,30],[-126.58,1.19,30],[-125.56,89.59,30],[-49.85,-89.76,30],[-49.13,0.86,30],[-49.28,89.65,30],[31.55,-89.33,30],[29.53,0.28,30],[29.20,89.69,30],[101.55,-0.07,30]],\
        "6": [[662.70,41.96,40],[662.41,144.54,40],[-1.63,-17.53,40],[-1.84,49.77,40],[-0.50,141.78,40],[1.29,244.84,40]],\
        "7": [[-197.22,-161.53,40],[-1.85,-238.09,40],[67.08,-1.04,35],[67.25,60.09,35],[-109.01,113.97,35],[-198.61,49.24,25],[-198.65,-36.34,25],[-151.27,48.35,25],[-100.17,-34.76,15],[-100.46,-63.77,15],[-101.47,-96.25,10],[-85.31,-111.70,10],[-73.35,-159.14,30],[-3.43,-159.27,30],[-4.05,-107.83,15],[-4.45,-64.86,20],[-4.79,57.83,35],[-101.62,53.08],[-3.78,-1.48],[-150.54,-35.13]],\
        "10": [[-44.76,-55.94,30],[96.00,-21.14,30],[96.84,68.01,30],[-46.40,127.21,30],[-99.79,19.70,30],[-38.44,65.96,30],[41.59,66.94,20],[41.08,30.14,20],[-47.38,19.22]]}

    def choose_start(self, index, frames, distance):
        start = super().choose_start(index, frames, distance)
        return start

    def turn_end(self, frame):
        return super().turn_end(frame)

    def skip_red_light(self, frames, start, end):
        start = super().skip_red_light(frames, start, end)
        return start

    def sample_Tjunctionframe(self, index, frames):
        start = index
        frame_range = len(frames)
        end = frame_range - 1
        instruction_args = []
        
        for i in range(start, frame_range):
            frame = frames[i]
            command = frame["command"]
            if command == self.direction_command_mapping[self.direction]:  # Endpoint of the sampling
                continue
            end = i
            if end - start > self.max_range:
                start = self.skip_red_light(frames, start, end)
            return start, end, instruction_args

        if end - start > self.max_range:
            start = self.skip_red_light(frames, start, end)
        return start, end, instruction_args

    def sample_frame(self, index, frames, town_loc):
        start = index
        frame_range = min(start+self.total_range,len(frames))
        end = frame_range - 1
        instruction_args = []
        
        for i in range(start, frame_range):
            frame = frames[i]
            command = frame["command"]
            if command != 4 :  # Endpoint of the sampling
                end = i
                return start, end, instruction_args
            for des_loc in town_loc:
                if math.sqrt(pow(des_loc[0]-frame["x"], 2)+pow(des_loc[1]-frame["y"], 2)) < self.sample_range:
                    end = i
                    return start, end, instruction_args    
        return start, end, instruction_args

    def add_result(self, results, index, frames, town_id):
        frame = frames[index]
        command = frame["command"]
        Tjuntion_loc = self.Tjunction_loc_mapping[str(town_id)][self.direction_index_mapping[self.direction]]
        if command == self.direction_command_mapping[self.direction]:
            for des_loc in Tjuntion_loc:
                if math.sqrt(pow(des_loc[0]-frame["x"], 2)+pow(des_loc[1]-frame["y"], 2)) < des_loc[2]:
                    start_frame, end_frame, instruction_args = self.sample_Tjunctionframe(index, frames)
                    if end_frame == None:
                        return index
                    if start_frame == None:
                        index = end_frame
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
        intersection_loc = self.intersection_loc_mapping[str(town_id)]
        if command != 4:
            return index
        for des_loc in intersection_loc:
            if math.sqrt(pow(des_loc[0]-frame["x"], 2)+pow(des_loc[1]-frame["y"], 2)) < self.sample_range:
                return index
        start_frame, end_frame, instruction_args = self.sample_frame(index, frames, intersection_loc)
        if end_frame == None:
            return index
        if start_frame == None:
            index = end_frame
            return index
        index = end_frame  # Move forward the index
        if end_frame - start_frame < 32:
            return index
        if random.random() > 0.915:
            result = {}
            result["start_frame"] = start_frame
            result["end_frame"] = end_frame
            result["instruction_args"] = instruction_args
            results.append(result)
        return index

    def process(self, data):
        results = []
        frames = data["data"]
        town_id = data["town_id"]
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames, town_id)
            index = index + 1
        return results

class IslandFalse():
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
        if random.random() > 0.985:
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
        if town_id == 3:
            return results
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1
        return results
