import os
import math
import random
import re
import json
from pathlib import Path
from collections import deque

import numpy as np
import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

DEBUG = int(os.environ.get("HAS_DISPLAY", 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x - r, y - r, x + r, y + r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0])  # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945])  # for carla 9.10

        self.debug = Plotter(debug_size)

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos["lat"], pos["lon"]])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        self.debug.clear()

        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(
                self.route[i][0] - self.route[i - 1][0]
            )
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self.route[i][1].value == 4)
            b = 255
            self.debug.dot(gps, self.route[i][0], (r, g, b))

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        self.debug.dot(gps, self.route[0][0], (0, 255, 0))
        self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        self.debug.dot(gps, gps, (0, 0, 255))
        self.debug.show()

        return self.route[1]

    def get_future_waypoints(self, num=10):
        res = []
        for i in range(min(num, len(self.route))):
            res.append(
                [self.route[i][0][0], self.route[i][0][1], self.route[i][1].value]
            )
        return res

class InstructionPlanner(object):
    def __init__(self, scenario_cofing_name = '', notice_light_switch = False):
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        lights_list = self._world.get_actors().filter("*traffic_light*")
        self._map = self._world.get_map()
        self._list_traffic_lights = []
        for light in lights_list:
            center, waypoints = self.get_traffic_light_waypoints(light)
            self._list_traffic_lights.append((light, center, waypoints))
        (
            self._list_traffic_waypoints,
            self._dict_traffic_lights,
        ) = self._gen_traffic_light_dict(self._list_traffic_lights)

        self.curr_command = None
        self.last_command = None
        self.curr_command_mislead = None
        self.prev_instruction_id = 60
        self.curr_instruction_id = 60
        self.prev_mislead_id = -1
        self.curr_mislead_id = -1
        with open(os.path.join(Path(__file__).parent.parent,'leaderboard/envs/instruction_dict.json')) as f:
            self.instruct_dict = json.load(f)
        self.instruct_dict["-1"] = [""]
        random.seed(int(re.search('\d+',scenario_cofing_name)[0]))
        self.prev_instruction = random.choice(self.instruct_dict['60'])
        self.curr_instruction = random.choice(self.instruct_dict['60'])
        self.prev_mislead = ''
        self.curr_mislead = ''
        self.last_target_point = np.array([0,0])
        self.last_target_point_mislead = np.array([0,0])
        self.highway_mapping = {"Town04":[[-487.84,361.47,2.84,44.26],[-19.73,18.43,-279.10,278.82],[94.88,333.41,-360.95,-398.73],[-376.92,-93.35,400.16,440.26],[-517.71,-478.04,37.87,319.51]], \
                                "Town05":[[-257.43,-217.99,-179.75,175.86],[184.14,218.66,-175.28,174.62],[-204.68,162.79,-217.05,-181.86],[-210.39,179.10,182.47,218.67]], \
                                "Town06":[[-302.75,625.72,-8.10,-26.18],[-278.19,651.95,35.46,54.74],[-286.53,649.00,135.70,155.15],[-323.99,647.78,236.17,254.11],[656.33,673.07,12.02,228.79],[-372.63,-359.37,13.01,230.13]] }
        self.all_junction_mapping = {"Town01":[[90.30,0.51,25,1],[156.93,1.09,25,1],[336.86,1.39,25,1],[337.33,326.93,40,3],[90.95,327.01,40,3],[92.37,196.73,30,0],[91.87,131.36,30,0],[92.17,57.97,25,0],[156.05,55.61,25,3],[335.12,57.68,25,2],[335.78,130.58,30,2],[336.32,196.97,30,2]], \
                                  "Town02":[[43.31,304.10,30,3],[-5.34,190.45,30,0],[192.52,189.99,30,2],[190.68,239.30,30,2],[134.06,238.50,30,3],[43.77,238.49,30,0],[43.51,190.68,30,1],[133.09,189.44,30,1]], \
                                  "Town03":[[3.93,-199.79,35,1],[236.43,0.77,30,2],[237.27,61.02,30,2],[-1.39,196.76,35,3],[151.53,-132.98,30,2],[149.59,-72.75,30,2],[80.90,-74.44,30,0],[148.68,-5.98,30,3],[78.58,-5.19,30,3],[169.12,64.11,30,1],[-226.23,-2.30,25],[-223.15,103.26,25],[83.79,-257.12,25],[157.84,-256.18,25],[-146.60,-1.44,25],[-84.86,133.58,25],[-2.82,132.36,25],[-81.72,-137.82,25],[2.44,-135.59,25],[83.89,-135.75,25],[85.39,-199.39,25],[153.65,-198.61,25]], \
                                  "Town04":[[257.15,-308.29,25,1],[256.30,-122.12,25,3],[128.78,-172.50,30,3],[61.36,-174.60,25,3],[15.01,-172.33,25,0],[205.67,-364.69,30,1],[393.50,-171.28,25,2],[381.09,-67.54,30,2],[203.01,-309.33,25],[202.12,-247.58,25],[200.61,-171.29,25],[256.94,-248.01,25],[256.49,-170.93,25],[313.26,-248.37,25],[15.56,-56.04,25],[90.57,39.99,25],[-16.52,105.38,25],[-83.45,5.42,25],[-6.80,-277.01,25],[-7.44,327.68,25],[75.02,6.16,25],[16.75,99.37,25],[-76.96,37.67,25],[-15.78,-50.79,25],[-383.98,1.90,25],[404.62,6.40,25]], \
                                  "Town05":[[34.01,-182.82,20,1],[40.02,-147.67,20,0],[153.47,-0.52,25,2],[40.85,142.48,25,0],[30.24,198.96,30,3],[-126.12,-137.57,20,1],[-124.06,148.97,25,3],[-268.82,-1.19,30,0],[-189.88,-90.40,25],[-189.49,0.79,25],[-190.41,89.65,25],[-127.13,-89.45,25],[-126.58,1.19,25],[-125.56,89.59,25],[-49.85,-89.76,25],[-49.13,0.86,25],[-49.28,89.65,25],[31.55,-89.33,25],[29.53,0.28,25],[29.20,89.69,25],[101.55,-0.07,25]], \
                                  "Town06":[[662.70,41.96,40,2],[662.41,144.54,40,2],[-1.63,-17.53,25],[-1.84,49.77,25],[-0.50,141.78,25],[1.29,244.84,25],[-137.44,-8.89,25],[132.52,38.19,25],[494.16,37.50,25],[-211.54,149.29,25],[469.45,137.07,25],[-111.28,237.35,25],[81.21,135.94,25],[98.99,236.78,25],[-140.42,42.52,25],[134.92,-15.95,25],[506.05,-12.70,25],[-211.91,236.81,25],[-111.96,148.93,25],[257.00,52.54,25],[549.61,52.18,25],[243.89,151.35,25]], \
                                  "Town07":[[-197.22,-161.53,40,0],[-1.85,-238.09,40,1],[67.08,-1.04,35,2],[67.25,60.09,35,2],[-109.01,113.97,35,3],[-198.61,49.24,25,0],[-198.65,-36.34,25,0],[-151.27,48.35,25,3],[-100.17,-0.26,15,0],[-100.17,-34.76,15,2],[-100.46,-63.77,15,0],[-101.47,-96.25,10,2],[-85.31,-111.70,10,0],[-73.35,-159.14,30,1],[-3.43,-159.27,30,2],[-4.05,-107.83,15,2],[-4.45,-64.86,20,2],[-4.79,57.83,35,3],[-101.62,53.08,25],[-3.78,-1.48,25],[-150.54,-35.13,25]], \
                                  "Town10HD":[[-44.76,-55.94,30,1],[96.00,21.14,30,2],[96.84,68.01,30,2],[-46.40,127.21,30,3],[-99.79,19.70,30,0],[-38.44,65.96,30,0],[41.59,66.94,20,3],[41.08,30.14,20,1],[-47.38,19.22,25]]}
        self.tjunction_mapping = {"Town01":[[90.30,0.51,25,1],[156.93,1.09,25,1],[336.86,1.39,25,1],[337.33,326.93,40,3],[90.95,327.01,40,3],[92.37,196.73,30,0],[91.87,131.36,30,0],[92.17,57.97,25,0],[156.05,55.61,25,3],[335.12,57.68,25,2],[335.78,130.58,30,2],[336.32,196.97,30,2]], \
                                  "Town02":[[43.31,304.10,30,3],[-5.34,190.45,30,0],[192.52,189.99,30,2],[190.68,239.30,30,2],[134.06,238.50,30,3],[43.77,238.49,30,0],[43.51,190.68,30,1],[133.09,189.44,30,1]], \
                                  "Town03":[[3.93,-199.79,35,1],[236.43,0.77,30,2],[237.27,61.02,30,2],[-1.39,196.76,35,3],[151.53,-132.98,30,2],[149.59,-72.75,30,2],[80.90,-74.44,30,0],[148.68,-5.98,30,3],[78.58,-5.19,30,3],[169.12,64.11,30,1]], \
                                  "Town04":[[257.15,-308.29,25,1],[256.30,-122.12,25,3],[128.78,-172.50,30,3],[61.36,-174.60,25,3],[15.01,-172.33,25,0],[205.67,-364.69,30,1],[393.50,-171.28,25,2],[381.09,-67.54,30,2]], \
                                  "Town05":[[34.01,-182.82,20,1],[40.02,-147.67,20,0],[153.47,-0.52,25,2],[40.85,142.48,25,0],[30.24,198.96,30,3],[-126.12,-137.57,20,1],[-124.06,148.97,25,3],[-268.82,-1.19,30,0]], \
                                  "Town06":[[662.70,41.96,40,2],[662.41,144.54,40,2]], \
                                  "Town07":[[-197.22,-161.53,40,0],[-1.85,-238.09,40,1],[67.08,-1.04,35,2],[67.25,60.09,35,2],[-109.01,113.97,35,3],[-198.61,49.24,25,0],[-198.65,-36.34,25,0],[-151.27,48.35,25,3],[-100.17,-0.26,15,0],[-100.17,-34.76,15,2],[-100.46,-63.77,15,0],[-101.47,-96.25,10,2],[-85.31,-111.70,10,0],[-73.35,-159.14,30,1],[-3.43,-159.27,30,2],[-4.05,-107.83,15,2],[-4.45,-64.86,20,2],[-4.79,57.83,35,3]], \
                                  "Town10HD":[[-44.76,-55.94,30,1],[96.00,21.14,30,2],[96.84,68.01,30,2],[-46.40,127.21,30,3],[-99.79,19.70,30,0],[-38.44,65.96,30,0],[41.59,66.94,20,3],[41.08,30.14,20,1]]}
        self.cross_mapping = {"Town03":[[-226.23,-2.30,25],[-223.15,103.26,25],[83.79,-257.12,25],[157.84,-256.18,25],[-146.60,-1.44,25],[-84.86,133.58,25],[-2.82,132.36,25],[-81.72,-137.82,25],[2.44,-135.59,25],[83.89,-135.75,25],[85.39,-199.39,25],[153.65,-198.61,25]], \
                              "Town04":[[203.01,-309.33,25],[202.12,-247.58,25],[200.61,-171.29,25],[256.94,-248.01,25],[256.49,-170.93,25],[313.26,-248.37,25]], \
                              "Town05":[[-189.88,-90.40,25],[-189.49,0.79,25],[-190.41,89.65,25],[-127.13,-89.45,25],[-126.58,1.19,25],[-125.56,89.59,25],[-49.85,-89.76,25],[-49.13,0.86,25],[-49.28,89.65,25],[31.55,-89.33,25],[29.53,0.28,25],[29.20,89.69,25],[101.55,-0.07,25]], \
                              "Town06":[[-1.63,-17.53,25],[-1.84,49.77,25],[-0.50,141.78,25],[1.29,244.84,25]], \
                              "Town07":[[-101.62,53.08,25],[-3.78,-1.48,25],[-150.54,-35.13,25]], \
                              "Town10HD":[[-47.38,19.22,25]]}
        self.leave_highway_mapping_right = {"Town04":[[15.56,-56.04,25],[90.57,39.99,25],[-16.52,105.38,25],[-83.45,5.42,25],[-6.80,-277.01,25]]}
        self.leave_highway_mapping_straight = {"Town04":[[-7.44,327.68,25]]}
        self.enter_highway_mapping_right = {"Town04":[[117.66,-11.77,30],[31.82,130.19,30],[-111.13,54.30,30],[-33.66,-87.58,30]],"Town06":[[529.11,59.11,30]]}
        self.enter_highway_mapping_left ={"Town06":[[-137.44,-8.89,25],[132.52,38.19,25],[494.16,37.50,25],[-211.54,149.29,25],[469.45,137.07,25],[-111.28,237.35,25],[72.75,236.78,25]]}
        self.enter_highway_mapping_straight ={"Town06":[[81.21,135.94,25],[222.93,59.38,10],[189.98,162,63,10]]}
        self.destination_instruction_mapping = {"0":2,"1":3,"4":7,"5":8,"6":9,"10":13,"11":14,"12":15,"16":19,"17":20,"18":21,"34":36,"35":37,"46":48,"47":49}
        self.destination_instruction_reverse = {"2":0,"3":1,"7":4,"8":5,"9":6,"13":10,"14":11,"15":12,"19":16,"20":17,"21":18,"36":34,"37":35,"48":46,"49":47}
        self.turn_instructionid_list = [0,1,4,5,6,10,11,12,16,17,18,46,47]
        self.lightTurn_instruction_mapping = {"10":[0,4,16],"11":[1,5,17],"12":[6,18]}
        self.follow_instruction_distance_mapping = {"38":40,"42":44,"43":45}
        self.destination_distance = 0
        self.trigger_distance = False
        self.frame_count = 0
        self.des_pos = np.array([0,0])
        self.left_lane_num = 0
        self.right_lane_num = 0
        self.notice = ''
        self.notice_freeze_time = 0
        self.notice_dis = 2
        self.notice_light_switch = notice_light_switch
        self.light_notice_text = ''
        self.light_notice_state = None
        self.town_id = ''
        self.notice_light_switch = True
        self.routes = None
        self.roundabout_instruction = ''

    def _gen_traffic_light_dict(self, traffic_lights_list):
        traffic_light_dict = {}
        waypoints_list = []
        for light, center, waypoints in traffic_lights_list:
            for waypoint in waypoints:
                traffic_light_dict[waypoint] = (light, center)
                waypoints_list.append(waypoint)
        return waypoints_list, traffic_light_dict

    def _find_closest_valid_traffic_light(self, loc, min_dis):
        wp = self._map.get_waypoint(loc)
        min_wp = None
        min_distance = min_dis
        for waypoint in self._list_traffic_waypoints:
            if waypoint.road_id != wp.road_id or waypoint.lane_id * wp.lane_id < 0:
                continue
            dis = loc.distance(waypoint.transform.location)
            if dis <= min_distance:
                min_distance = dis
                min_wp = waypoint
        if min_wp is None:
            return None
        else:
            return self._dict_traffic_lights[min_wp][0]

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = (
            math.cos(math.radians(angle)) * point.x
            - math.sin(math.radians(angle)) * point.y
        )
        y_ = (
            math.sin(math.radians(angle)) * point.x
            + math.cos(math.radians(angle)) * point.y
        )
        return carla.Vector3D(x_, y_, point.z)

    def get_traffic_light_waypoints(self, traffic_light):
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(
            -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
        )  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if (
                not ini_wps
                or ini_wps[-1].road_id != wpx.road_id
                or ini_wps[-1].lane_id != wpx.lane_id
            ):
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps

    def _generate_roundabout_instruction(self, curr_point, routes):
        i = 0
        angle_list = [np.pi/2,np.pi,np.pi*3/2]
        origin_point = np.array([0,0])
        match_index = 0
        index_instruction_mapping = [22,23,24]
        for point in routes:
            route = point[0]
            if i == 0: # Skip routes[0]
                i += 1
                continue
            waypoint = np.array([route[0],route[1]])
            if np.linalg.norm(waypoint-origin_point) > 45:
                angle = self.vectorangle(curr_point, waypoint)
                for index in range(3):
                    match_angle = angle_list[index]
                    if abs(angle-match_angle) < np.pi/4:
                        match_index = index_instruction_mapping[index]
                        roundabout_instruction = random.choice(self.instruct_dict[str(match_index)])
                        return roundabout_instruction
        return ''


    def _update_instruct(self, instruction_id, tick_data, dis_on=True):
        if np.linalg.norm(self.last_target_point-tick_data['target_point']) > 3 or instruction_id in self.turn_instructionid_list: # Update instruct when waypoint changes or command changes to turn command
            if instruction_id != self.curr_instruction_id:
                if str(instruction_id) in self.destination_instruction_mapping and dis_on and random.random() > 0.5 and self.curr_instruction_id not in self.turn_instructionid_list: # Convert instruction to instruction with distance
                    self.des_pos = np.array([tick_data["next_waypoint"][0],tick_data["next_waypoint"][1]])
                    curr_pos = np.array([tick_data["gps"][0],tick_data["gps"][1]])
                    curr_dis = np.linalg.norm(curr_pos-self.des_pos)
                    self.destination_distance = min(8 + 12*random.random(), max(curr_dis-5, 2))
                    if self.destination_distance >= 8: # Instruction distance must be bigger than 8 (7.5 is the waypoint update distance)
                        dis_instruction_id = self.destination_instruction_mapping[str(instruction_id)]
                    else:
                        self.destination_distance = 0
                        dis_instruction_id = instruction_id
                    if self.curr_instruction_id == 34 or self.curr_instruction_id == 35: # The last change lane waypoint need to be converted to follow waypoint
                        self.prev_instruction_id = 38
                        self.curr_instruction_id = instruction_id
                        self.prev_instruction = random.choice(self.instruct_dict['38'])
                        self.curr_instruction = random.choice(self.instruct_dict[str(dis_instruction_id)])
                    elif self.curr_instruction_id in self.turn_instructionid_list and self.prev_instruction_id in self.turn_instructionid_list: # Avoid giving turn instructions continously
                        self.prev_instruction_id = 38
                        self.curr_instruction_id = instruction_id
                        self.prev_instruction = random.choice(self.instruct_dict['38'])
                        self.curr_instruction = random.choice(self.instruct_dict[str(dis_instruction_id)])
                    else:
                        self.prev_instruction_id = self.curr_instruction_id
                        self.curr_instruction_id = instruction_id
                        self.prev_instruction = self.curr_instruction
                        self.curr_instruction = random.choice(self.instruct_dict[str(dis_instruction_id)])
                else:
                    if instruction_id in self.turn_instructionid_list and str(self.curr_instruction_id) in self.follow_instruction_distance_mapping.keys() and random.random() > 0.5 and dis_on: # Follow for a distance
                        if random.random() > 0.8:
                            self.destination_distance = 0
                            self.prev_instruction_id = 64
                            self.curr_instruction_id = instruction_id
                            x_distance = tick_data["target_point"][0]
                            y_distance = tick_data["target_point"][1]
                            if x_distance < 0:
                                navigation_direction = "left"
                            else:
                                navigation_direction = "right"
                            instruction_text = random.choice(self.instruct_dict[str(self.prev_instruction_id)]).replace("[x]", str(int(abs(y_distance)))).replace("[y]", str(int(abs(x_distance)))).replace("left/right", navigation_direction)
                            self.prev_instruction = instruction_text
                            self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                        else:
                            self.destination_distance = 0
                            self.prev_instruction_id = self.follow_instruction_distance_mapping[str(self.curr_instruction_id)]
                            self.curr_instruction_id = instruction_id
                            curr_pos = np.array([tick_data["gps"][0],tick_data["gps"][1]])
                            next_pos = np.array([tick_data["next_waypoint"][0],tick_data["next_waypoint"][1]])
                            follow_distance = np.linalg.norm(curr_pos-next_pos)
                            self.prev_instruction = random.choice(self.instruct_dict[str(self.prev_instruction_id)]).replace("[x]", str(int(follow_distance)))
                            self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                    elif self.curr_instruction_id == 39 and random.random() > 0.5 and dis_on:
                        self.destination_distance = 0
                        self.prev_instruction_id = 41
                        self.curr_instruction_id = instruction_id
                        curr_pos = np.array([tick_data["gps"][0],tick_data["gps"][1]])
                        next_pos = np.array([tick_data["next_waypoint"][0],tick_data["next_waypoint"][1]])
                        follow_distance = np.linalg.norm(curr_pos-next_pos)
                        self.prev_instruction = random.choice(self.instruct_dict['41']).replace("[x]", str(int(follow_distance)))
                        self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                    elif self.curr_instruction_id == 34 or self.curr_instruction_id == 35: # The last change lane waypoint need to be converted to follow waypoint
                        curr_pos = np.array([tick_data["gps"][0],tick_data["gps"][1]])
                         # Convert last change lane waypoint to follow waypoint
                        self.destination_distance = 0
                        self.prev_instruction_id = 38
                        self.curr_instruction_id = instruction_id
                        self.prev_instruction = random.choice(self.instruct_dict['38'])
                        self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                    elif self.curr_instruction_id in self.turn_instructionid_list and self.prev_instruction_id in self.turn_instructionid_list: # Avoid giving turn instructions continously
                        self.destination_distance = 0
                        self.prev_instruction_id = 38
                        self.curr_instruction_id = instruction_id
                        self.prev_instruction = random.choice(self.instruct_dict['38'])
                        self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                    else:
                        self.destination_distance = 0
                        self.prev_instruction_id = self.curr_instruction_id
                        self.curr_instruction_id = instruction_id
                        self.prev_instruction = self.curr_instruction
                        self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                curr_location = carla.Location(x=tick_data["gps"][1], y=-tick_data["gps"][0], z=0.00)
                if not self.trigger_distance and self.prev_instruction_id in self.turn_instructionid_list and self._find_closest_valid_traffic_light(curr_location, min_dis=50) is not None and random.random() > 0.3: # Convert a normal turn instruction to "turn at traffic light" instruction
                    for k,v in self.lightTurn_instruction_mapping.items():
                        if self.prev_instruction_id in v:
                            self.prev_instruction = random.choice(self.instruct_dict[k])
                self.trigger_distance = False
            elif np.linalg.norm(self.last_target_point-tick_data['target_point']) > 3: # Waypoint changes
                if self.curr_instruction_id == 34 or self.curr_instruction_id == 35: # Car is changing lane
                    self.prev_instruction_id = self.curr_instruction_id
                    self.curr_instruction_id = instruction_id
                    self.prev_instruction = self.curr_instruction
                    instruction = random.choice(self.instruct_dict[str(instruction_id)])
                    while instruction == self.curr_instruction: # Make sure change lane instructions are different every time
                        instruction = random.choice(self.instruct_dict[str(instruction_id)])
                    self.curr_instruction = instruction
                    self.destination_distance = 0
                elif self.curr_instruction_id in self.turn_instructionid_list and self.prev_instruction_id in self.turn_instructionid_list: # Avoid giving turn instructions continously
                    self.destination_distance = 0
                    self.prev_instruction_id = 38
                    self.curr_instruction_id = instruction_id
                    self.prev_instruction = random.choice(self.instruct_dict['38'])
                    self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                elif self.curr_instruction_id in self.destination_instruction_mapping.values(): # Avoid giving instructin with distance continuously
                    self.curr_instruction_id = self.destination_instruction_reverse[str(self.curr_instruction_id)]
                    self.prev_instruction = self.curr_instruction
                    self.curr_instruction = random.choice(self.instruct_dict[str(self.curr_instruction_id)])
                    self.destination_distance = 0
                elif str(self.curr_instruction_id) in self.follow_instruction_distance_mapping.keys() and random.random() > 0.8 and dis_on: # Navigation instruct
                        self.destination_distance = 0
                        self.prev_instruction_id = 64
                        self.curr_instruction_id = instruction_id
                        x_distance = tick_data["target_point"][0]
                        y_distance = tick_data["target_point"][1]
                        if x_distance < 0:
                            navigation_direction = "left"
                        else:
                            navigation_direction = "right"
                        instruction_text = random.choice(self.instruct_dict[str(self.prev_instruction_id)]).replace("[x]", str(int(abs(y_distance)))).replace("[y]", str(int(abs(x_distance)))).replace("left/right", navigation_direction)
                        self.prev_instruction = instruction_text
                        self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                else:
                    self.prev_instruction_id = self.curr_instruction_id
                    self.curr_instruction_id = instruction_id
                    self.prev_instruction = self.curr_instruction
                    self.curr_instruction = random.choice(self.instruct_dict[str(instruction_id)])
                    self.destination_distance = 0
                self.trigger_distance = False
        if self.destination_distance != 0:
            self.curr_instruction = self.curr_instruction.replace("[x]", str(int(self.destination_distance)))
        self.last_target_point = tick_data['target_point']
        if self.town_id == "Town03" and self.routes != None:
            curr_point = np.array([tick_data["gps"][0],tick_data["gps"][1]])
            origin_point = np.array([0,0])
            if np.linalg.norm(curr_point-origin_point) <= 30: # Roundabout range
                if self.roundabout_instruction == '':
                    self.roundabout_instruction = self._generate_roundabout_instruction(curr_point, self.routes)
                self.prev_instruction = self.roundabout_instruction

    def _update_mislead(self, mislead_id, tick_data):
        curr_gps = np.array([tick_data["gps"][0],tick_data["gps"][1]])
        curr_location = carla.Location(x=curr_gps[1], y=-curr_gps[0], z=0.00)
        curr_waypoint = self._map.get_waypoint(curr_location)
        self.left_lane_num = self.get_left_nums(curr_waypoint)
        self.right_lane_num = self.get_right_nums(curr_waypoint)
        if np.linalg.norm(self.last_target_point_mislead-tick_data['target_point']) > 3:
            self.prev_mislead_id = self.curr_mislead_id
            self.curr_mislead_id = mislead_id
            self.prev_mislead = self.curr_mislead
            self.curr_mislead = random.choice(self.instruct_dict[str(mislead_id)])
        if self.prev_mislead_id == 34:
            if self.get_left_nums(curr_waypoint) != 0:
                self.prev_mislead_id = -1
                self.prev_mislead = random.choice(self.instruct_dict[str(self.prev_mislead_id)])
        elif self.prev_mislead_id == 35:
            if self.get_right_nums(curr_waypoint) != 0:
                self.prev_mislead_id = -1
                self.prev_mislead = random.choice(self.instruct_dict[str(self.prev_mislead_id)])
        self.last_target_point_mislead = tick_data['target_point']

    def command2instruct(self, town_id, tick_data, routes=None, dis_on=True):
        self.town_id = town_id
        self.routes = routes
        self.frame_count = self.frame_count + 1
        if self.frame_count == 64 and self.prev_instruction_id == 60:
            self.prev_instruction_id = 38
            self.prev_instruction = random.choice(self.instruct_dict['38'])
        instruction_id = None
        command = tick_data["next_command"]
        gps_pos = tick_data["next_waypoint"]
        if self.destination_distance != 0:
            curr_pos = np.array([tick_data["gps"][0],tick_data["gps"][1]])
            curr_dis = np.linalg.norm(curr_pos-self.des_pos)
            if abs(curr_dis-self.destination_distance) < 2:
                if self.last_command not in [1,2,3]:
                    self.prev_instruction = self.curr_instruction
                    self.destination_distance = 0
                    self.trigger_distance = True # Avoid distance instruction converting to traffic light instruction
                else:
                    self.curr_instruction = random.choice(self.instruct_dict[str(self.curr_instruction_id)])
                    self.destination_distance = 0
                    self.trigger_distance = False
        if self.curr_command != command:
            self.last_command = self.curr_command
            self.curr_command = command
        if self.curr_command == 1:
            if town_id in self.enter_highway_mapping_left.keys():
                for enter_highway_loc in self.enter_highway_mapping_left[town_id]:
                    if math.sqrt(pow(-enter_highway_loc[1]-gps_pos[0],2)+pow(enter_highway_loc[0]-gps_pos[1],2))<enter_highway_loc[2]:
                        instruction_id = 46
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            if town_id in self.tjunction_mapping.keys():
                for tjunction_loc in self.tjunction_mapping[town_id]:
                    if math.sqrt(pow(-tjunction_loc[1]-gps_pos[0],2)+pow(tjunction_loc[0]-gps_pos[1],2))<tjunction_loc[2]:
                        instruction_id = 16
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            if town_id in self.cross_mapping.keys():
                for cross_loc in self.cross_mapping[town_id]:
                    if math.sqrt(pow(-cross_loc[1]-gps_pos[0],2)+pow(cross_loc[0]-gps_pos[1],2))<cross_loc[2]:
                        instruction_id = 4
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            instruction_id = 0
            self._update_instruct(instruction_id, tick_data, dis_on)
            return self.prev_instruction
        elif self.curr_command == 2:
            if town_id in self.enter_highway_mapping_right.keys():
                for enter_highway_loc in self.enter_highway_mapping_right[town_id]:
                    if math.sqrt(pow(-enter_highway_loc[1]-gps_pos[0],2)+pow(enter_highway_loc[0]-gps_pos[1],2))<enter_highway_loc[2]:
                        instruction_id = 46
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            if town_id in self.leave_highway_mapping_right.keys():
                for leave_highway_loc in self.leave_highway_mapping_right[town_id]:
                    if math.sqrt(pow(-leave_highway_loc[1]-gps_pos[0],2)+pow(leave_highway_loc[0]-gps_pos[1],2))<leave_highway_loc[2]:
                        instruction_id = 47
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            if town_id in self.tjunction_mapping.keys():
                for tjunction_loc in self.tjunction_mapping[town_id]:
                    if math.sqrt(pow(-tjunction_loc[1]-gps_pos[0],2)+pow(tjunction_loc[0]-gps_pos[1],2))<tjunction_loc[2]:
                        instruction_id = 17
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            if town_id in self.cross_mapping.keys():
                for cross_loc in self.cross_mapping[town_id]:
                    if math.sqrt(pow(-cross_loc[1]-gps_pos[0],2)+pow(cross_loc[0]-gps_pos[1],2))<cross_loc[2]:
                        instruction_id = 5
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            instruction_id  = 1
            self._update_instruct(instruction_id, tick_data, dis_on)
            return self.prev_instruction
        elif self.curr_command == 3:
            if town_id in self.enter_highway_mapping_straight.keys():
                for enter_highway_loc in self.enter_highway_mapping_straight[town_id]:
                    if math.sqrt(pow(-enter_highway_loc[1]-gps_pos[0],2)+pow(enter_highway_loc[0]-gps_pos[1],2))<enter_highway_loc[2]:
                        instruction_id = 46
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            if town_id in self.leave_highway_mapping_straight.keys():
                for leave_highway_loc in self.leave_highway_mapping_straight[town_id]:
                    if math.sqrt(pow(-leave_highway_loc[1]-gps_pos[0],2)+pow(leave_highway_loc[0]-gps_pos[1],2))<leave_highway_loc[2]:
                        instruction_id = 47
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            if town_id in self.tjunction_mapping.keys():
                for tjunction_loc in self.tjunction_mapping[town_id]:
                    if math.sqrt(pow(-tjunction_loc[1]-gps_pos[0],2)+pow(tjunction_loc[0]-gps_pos[1],2))<tjunction_loc[2]:
                        instruction_id = 18
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            instruction_id  = 6
            self._update_instruct(instruction_id, tick_data, dis_on)
            return self.prev_instruction
        elif self.curr_command == 4:
            if town_id in self.highway_mapping.keys():
                for highway_range in self.highway_mapping[town_id]:
                    if gps_pos[1]>highway_range[0] and gps_pos[1]<highway_range[1] and -1*gps_pos[0]>highway_range[2] and -1*gps_pos[0]<highway_range[3]:
                        instruction_id = 39
                        self._update_instruct(instruction_id, tick_data, dis_on)
                        return self.prev_instruction
            instruction_id = random.choice([38,42,43])
            self._update_instruct(instruction_id, tick_data, dis_on)
            return self.prev_instruction
        elif self.curr_command == 5:
            instruction_id  = 34
            self._update_instruct(instruction_id, tick_data, dis_on)
            return self.prev_instruction
        elif self.curr_command == 6:
            instruction_id  = 35
            self._update_instruct(instruction_id, tick_data, dis_on)
            return self.prev_instruction
        else:
            instruction_id = 63
            self._update_instruct(instruction_id, tick_data, dis_on)
            return self.prev_instruction

    def pos2notice(self, sampled_scenarios, tick_data):
        if sampled_scenarios == []:
            return self.notice
        if self.notice_freeze_time >0:
            self.notice_freeze_time = self.notice_freeze_time - 1
            return self.notice
        pos = np.array([tick_data["gps"][1],-tick_data["gps"][0]])
        scenarios_min_dis = np.array(float('inf'))
        scenarios_coordinates_dict = {}
        for scenario in sampled_scenarios:
            if scenario['name'] not in scenarios_coordinates_dict:
                scenarios_coordinates_dict[scenario['name']] = np.array([[scenario['trigger_position']['x'],scenario['trigger_position']['y']]])
            else:
                scenarios_coordinates_dict[scenario['name']] = np.append(scenarios_coordinates_dict[scenario['name']],np.array([[scenario['trigger_position']['x'],scenario['trigger_position']['y']]]),axis=0)

        for scenario,coordinates in scenarios_coordinates_dict.items():
            dis = np.linalg.norm(coordinates - pos,axis=1,keepdims=True)
            scenario_min_dis = dis[np.argmin(dis)]
            if scenario_min_dis < scenarios_min_dis :
                scenarios_min_dis = scenario_min_dis
                match_scenario = scenario
        if scenarios_min_dis[0] > self.notice_dis:
            self.notice_freeze_time = 0
            self.notice = ''
            return self.notice
        elif match_scenario == "Scenario3":
            self.notice_freeze_time = 80
            self.notice = random.choice(self.instruct_dict['50'])
            return self.notice
        elif match_scenario == "Scenario4":
            self.notice_freeze_time = 80
            self.notice = random.choice(self.instruct_dict['51'])
            return self.notice
        elif match_scenario == "Scenario2" or match_scenario == "Scenario5":
            self.notice_freeze_time = 80
            self.notice = random.choice(self.instruct_dict['52'])
            return self.notice
        elif match_scenario == "Scenario8" or match_scenario == "Scenario9":
            self.notice_freeze_time = 80
            self.notice = random.choice(self.instruct_dict['53'])
            return self.notice
        elif match_scenario == "Scenario7":
            self.notice_freeze_time = 80
            self.notice = random.choice(self.instruct_dict['54'])
            return self.notice
        elif match_scenario == "Scenario1":
            self.notice_freeze_time = 80
            self.notice = random.choice(self.instruct_dict['55'])
            return self.notice
        else:
            return self.notice

    def diff_angle(self, a, b):
        result = min(abs(a - b), np.pi * 2 - abs(a - b))
        return result

    def command2mislead(self, town_id, tick_data):
        mislead_id = None
        mislead_id_list = []
        command = tick_data["next_command"]
        gps_pos = tick_data["next_waypoint"]
        theta = tick_data["measurements"][2]
        location = carla.Location(x=gps_pos[1], y=-gps_pos[0], z=0.00)
        waypoint = self._map.get_waypoint(location)
        self.curr_command_mislead = command
        if self.curr_command_mislead in [1,2,3]:
            for tjunction_loc in self.tjunction_mapping[town_id]:
                if math.sqrt(pow(-tjunction_loc[1]-gps_pos[0],2)+pow(tjunction_loc[0]-gps_pos[1],2))<tjunction_loc[2]:
                    target_angle_left = (tjunction_loc[3] * np.pi / 2) % (np.pi * 2)
                    target_angle_right = (tjunction_loc[3] * np.pi / 2 + np.pi) % (np.pi * 2)
                    if self.diff_angle(theta, target_angle_left) < np.pi / 9:
                        mislead_id = 16
                        mislead_id_list.append(mislead_id)
                    if self.diff_angle(theta, target_angle_right) < np.pi / 9:
                        mislead_id = 17
                        mislead_id_list.append(mislead_id)
            mislead_id = 34
            mislead_id_list.append(mislead_id)
            mislead_id = 35
            mislead_id_list.append(mislead_id)
        if self.curr_command_mislead == 4:
            if town_id in ["Town01","Town02","Town07"]:
                mislead_id = 34
                mislead_id_list.append(mislead_id)
                mislead_id = 35
                mislead_id_list.append(mislead_id)
            if self.get_left_nums(waypoint) == 0:
                mislead_id = 34
                mislead_id_list.append(mislead_id)
            if self.get_right_nums(waypoint) == 0:
                mislead_id = 35
                mislead_id_list.append(mislead_id)
            if town_id in self.all_junction_mapping.keys():
                turn_mislead_flag = True
                for cross_loc in self.all_junction_mapping[town_id]:
                    target_theta = self.azimuthangle(cross_loc[0], -cross_loc[1], gps_pos[1], gps_pos[0])
                    if self.diff_angle(target_theta, theta) < np.pi/6:
                        if math.sqrt(pow(-cross_loc[1]-gps_pos[0],2)+pow(cross_loc[0]-gps_pos[1],2)) < 100:
                            turn_mislead_flag = False
                if turn_mislead_flag:
                    mislead_id = 4
                    mislead_id_list.append(mislead_id)
                    mislead_id = 5
                    mislead_id_list.append(mislead_id)
        if town_id not in ["Town03"]:
            mislead_id_list.append(22)
            mislead_id_list.append(23)
            mislead_id_list.append(24)
        if town_id not in ["Town04","Town05","Town06"]:
            mislead_id_list.append(39)
            mislead_id_list.append(46)
            mislead_id_list.append(47)
        if mislead_id_list != []:
            mislead_id = random.choice(mislead_id_list)
        else:
            mislead_id = -1
        self._update_mislead(mislead_id, tick_data)
        return self.prev_mislead

    def traffic_notice(self, tick_data):
        light = self._find_closest_valid_traffic_light(
            self._vehicle.get_location(), min_dis=50
        )
        if not self.notice_light_switch :
            return self.light_notice_text
        if light is not None:
            pos = np.array([tick_data["gps"][1],-tick_data["gps"][0]])
            light_pos = [light.get_transform().location.x,light.get_transform().location.y]
            light_distance = np.linalg.norm(light_pos - pos)
            if light_distance < 40:
                if light.state == carla.TrafficLightState.Green:
                    if self.light_notice_state != light.state:
                        self.light_notice_state = light.state
                        self.light_notice_text = random.choice(self.instruct_dict['58'])
                elif light.state == carla.TrafficLightState.Red:
                    if self.light_notice_state != light.state:
                        self.light_notice_state = light.state
                        self.light_notice_text = random.choice(self.instruct_dict['57'])
                elif light.state == carla.TrafficLightState.Yellow:
                    if self.light_notice_state != light.state:
                        self.light_notice_state = light.state
                        self.light_notice_text = random.choice(self.instruct_dict['59'])
                else:
                    self.light_notice_text = ''
                return self.light_notice_text
            else:
                self.light_notice_text = ''
                return self.light_notice_text
        else:
            self.light_notice_text = ''
            return self.light_notice_text

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

    def azimuthangle(self, x1, y1, x2, y2):
        x = math.atan2(y2-y1,x2-x1)
        if x < 0:
            x = 2*np.pi + x
        x = 2*np.pi - x
        x = (x + np.pi/2) % (np.pi*2)
        return x

    def vectorangle(self, v1, v2):
        r = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)))
        deg = r

        a1 = np.array([*v1, 0])
        a2 = np.array([*v2, 0])

        a3 = np.cross(a1, a2)

        if np.sign(a3[2]) > 0:
            deg = np.pi*2 - deg

        return deg

class MultiInsturctionsPlanner(InstructionPlanner):
    def __init__(self, global_plan, scenario_cofing_name = '', notice_light_switch = False):
        super().__init__(scenario_cofing_name, notice_light_switch)
        self._global_plan = global_plan
        self._route_planner = RoutePlanner(5.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.is_combine = False
        self.combine_instruction = ''
        self.route_length = None
        self.prob = 0.9 # Probobility of combining instruction (reverse)

    def command2multiInstruct(self, town_id, tick_data, routes=None):
        next_wp, next_cmd = self._route_planner.run_step(tick_data["gps"])
        combine_prob = random.random()
        if combine_prob > self.prob or self.is_combine:
            generate_instruction = self.command2instruct(town_id, tick_data, dis_on=False)
        else:
            generate_instruction = self.command2instruct(town_id, tick_data, routes)
        if not self.is_combine:
            self.combine_instruction = generate_instruction
        if self.route_length != len(self._route_planner.route):
            self.route_length = len(self._route_planner.route)
            self.combine_instruction = generate_instruction
            if town_id == "Town03" and np.linalg.norm(np.array([tick_data["gps"][0],tick_data["gps"][1]])-np.array([0,0])) < 100:
                self.is_combine = False
                return self.combine_instruction
            if combine_prob > self.prob: # Combine next one or two instructions
                self.is_combine = True
                if random.random() > 0.5:
                    combine_num = 1
                else:
                    combine_num = 2
                for i in range(combine_num):
                    input_data = {}
                    gps = next_wp
                    if len(self._route_planner.route) > 2:
                        next_wp, next_cmd = self._route_planner.run_step(gps)
                        input_data["gps"] = gps
                        input_data["next_waypoint"] = next_wp
                        input_data["next_command"] = next_cmd.value
                        input_data["measurements"] = [gps[0], gps[1]]
                        input_data["target_point"] = next_wp
                        prev_id = self.prev_instruction_id
                        generate_instruction = self.command2instruct(town_id, input_data, dis_on=False)
                        if prev_id != self.prev_instruction_id:
                            self.combine_instruction += ';' + generate_instruction
                self.route_length = len(self._route_planner.route)
            else:
                self.is_combine = False
        return self.combine_instruction
