import random
import math

class Notice():
    def __init__(self) -> None:
        self.notice_type = []

    def add_result(self, results, frames, transform):
        frame_num = len(frames)
        distance_min = float('inf')
        for index in range(frame_num):
            frame = frames[index]
            distance = math.sqrt(pow(transform["transform"]["x"]-frame["x"], 2)+pow(transform["transform"]["y"]-frame["y"], 2))
            if distance < distance_min:
                distance_min = distance
                index_min = index

        if distance_min < 2:
            result = {}
            result["frame_id"] = index_min
            result["instruction_args"] = []
            if not result in results:
                results.append(result)

    def choose_scenario_configs(self, scenario, scenarios_event_configs):
        for notice in self.notice_type:
            if scenario["scenario_type"] == notice:
                scenario_event = scenario["available_event_configurations"]
                scenarios_event_configs.append(scenario_event)

    def process(self, data, scenarios, town_str):
        results = []
        frames = data["data"]
        if town_str == "10":
            town_str = "10HD"
        town_str = "Town" + town_str
        scenarios_event_configs = []
        scenarios_data = scenarios["available_scenarios"][0][town_str]
        for scenario in scenarios_data:
            self.choose_scenario_configs(scenario, scenarios_event_configs)

        for scenario_event_configs in scenarios_event_configs:
            for transform in scenario_event_configs:
                self.add_result(results, frames, transform)

        return results

class Notice01(Notice):
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario3"]

class Notice02(Notice):
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario4"]

class Notice03(Notice):
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario2","Scenario5"]

class Notice04(Notice):
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario8","Scenario9"]

class Notice05(Notice):
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario7"]

class Notice06(Notice):
    def __init__(self):
        super().__init__()
        self.notice_type = ["Scenario1"]

class Notice07():
    def __init__(self):
        self.scenario_loc = [[-211.35,-93.20],[-263.20,-69.99],[-211.31,-2.16],[-260.53,5.01],[-209.38,85.59],[-260.65,91.40]]

    def add_result(self,results,index,frames):
        frame = frames[index]
        for loc in self.scenario_loc:
            if math.sqrt(pow(loc[0]-frame["x"], 2)+pow(loc[1]-frame["y"], 2)) < 2:
                result = {}
                result["frame_id"] = index
                result["instruction_args"] = []
                results.append(result)
                index = min(index+128, len(frames)-1)
                return index
        return index

    def process(self, data, scenarios, town_str):
        results = []
        if town_str != "05":
            return results

        frames = data["data"]
        frame_num = len(frames)
        index = 15
        while index < frame_num :
            index = self.add_result(results, index, frames)
            index = index + 1

        return results

class Notice08():
    def __init__(self, light_status):
        self.light_status = light_status
        self.light_status_mapping = ['red','yellow','green']
        self.light_dis_town_mapping = {"01": 25,"02": 25,"03": 45,"04": 25,"05": 45,"06": 25,"07": 25,"10": 45}

    def add_result(self, results, index, frames, town_str):
        frame = frames[index]
        affected_light_id = frame["affected_light_id"]
        if affected_light_id == -1:
            return index
        if not str(affected_light_id) in frame["actors_data"].keys():
            return index
        light_loc = frame["actors_data"][str(affected_light_id)]["loc"]
        light_dis = math.ceil(math.sqrt(pow(light_loc[0]-frame["x"], 2)+pow(light_loc[1]-frame["y"], 2)))
        if light_dis > self.light_dis_town_mapping[town_str] + random.random()*10:
            return index
        light_state = frame["actors_data"][str(affected_light_id)]["sta"]
        if self.light_status_mapping[light_state] == self.light_status:
            result = {}
            result["frame_id"] = index
            result["instruction_args"] = []
            results.append(result)
            index = min(index+128, len(frames)-1)
            return index
        return index

    def process(self, data, scenarios, town_str):
        results = []
        frames = data["data"]
        frame_num = len(frames)
        index = 15
        while index < frame_num:
            index = self.add_result(results, index, frames, town_str)
            index = index +1
        return results

