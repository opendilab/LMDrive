import os
import random

routes = {}
routes[
    "training_routes/routes_town01_short.xml"
] = "scenarios/town01_all_scenarios.json"
routes["training_routes/routes_town01_tiny.xml"] = "scenarios/town01_all_scenarios.json"
routes[
    "training_routes/routes_town02_short.xml"
] = "scenarios/town02_all_scenarios.json"
routes["training_routes/routes_town02_tiny.xml"] = "scenarios/town02_all_scenarios.json"
routes[
    "training_routes/routes_town03_short.xml"
] = "scenarios/town03_all_scenarios.json"
routes["training_routes/routes_town03_tiny.xml"] = "scenarios/town03_all_scenarios.json"
routes[
    "training_routes/routes_town04_short.xml"
] = "scenarios/town04_all_scenarios.json"
routes["training_routes/routes_town04_tiny.xml"] = "scenarios/town04_all_scenarios.json"
routes[
    "training_routes/routes_town05_short.xml"
] = "scenarios/town05_all_scenarios.json"
routes["training_routes/routes_town05_tiny.xml"] = "scenarios/town05_all_scenarios.json"
routes["training_routes/routes_town05_long.xml"] = "scenarios/town05_all_scenarios.json"
routes[
    "training_routes/routes_town06_short.xml"
] = "scenarios/town06_all_scenarios.json"
routes["training_routes/routes_town06_tiny.xml"] = "scenarios/town06_all_scenarios.json"
routes[
    "training_routes/routes_town07_short.xml"
] = "scenarios/town07_all_scenarios.json"
routes["training_routes/routes_town07_tiny.xml"] = "scenarios/town07_all_scenarios.json"
routes[
    "training_routes/routes_town10_short.xml"
] = "scenarios/town10_all_scenarios.json"
routes["training_routes/routes_town10_tiny.xml"] = "scenarios/town10_all_scenarios.json"
routes[
    "additional_routes/routes_town01_long.xml"
] = "scenarios/town01_all_scenarios.json"
routes[
    "additional_routes/routes_town02_long.xml"
] = "scenarios/town02_all_scenarios.json"
routes[
    "additional_routes/routes_town03_long.xml"
] = "scenarios/town03_all_scenarios.json"
routes[
    "additional_routes/routes_town04_long.xml"
] = "scenarios/town04_all_scenarios.json"
routes[
    "additional_routes/routes_town06_long.xml"
] = "scenarios/town06_all_scenarios.json"

ip_ports = []

for port in range(2000, 2008, 2):
    ip_ports.append(("localhost", port, port + 500))


carla_seed = 2000
traffic_seed = 2000


def generate_script(
    ip, port, tm_port, route, scenario, carla_seed, traffic_seed, save_path
):
    lines = []
    lines.append("export HOST=%s\n" % ip)
    lines.append("export PORT=%d\n" % port)
    lines.append("export TM_PORT=%d\n" % tm_port)
    lines.append("export ROUTES=${LEADERBOARD_ROOT}/data/%s\n" % route)
    lines.append("export SCENARIOS=${LEADERBOARD_ROOT}/data/%s\n" % scenario)
    lines.append("export CARLA_SEED=%d\n" % port)
    lines.append("export TRAFFIC_SEED=%d\n" % port)
    lines.append("export TEAM_CONFIG=${YAML_ROOT}/auto_agent.yaml\n")
    lines.append("export SAVE_PATH=${DATA_ROOT}/%s/data\n" % save_path)
    lines.append(
        "export CHECKPOINT_ENDPOINT=${DATA_ROOT}/%s/results/%s.json\n"
        % (save_path, route.split("/")[1].split(".")[0])
    )
    lines.append("\n")
    base = open("base_script.sh").readlines()

    for line in lines:
        base.insert(13, line)

    return base


for i in range(4):
    if not os.path.exists("bashs"):
        os.mkdir("bashs")
    os.mkdir("bashs/sub-%d" % i)
    for route in routes:
        ip, port, tm_port = ip_ports[i]
        script = generate_script(
            ip,
            port,
            tm_port,
            route,
            routes[route],
            carla_seed,
            traffic_seed,
            'sub-%d' % i,
        )
        fw = open(
            "bashs/sub-%d/%s.sh" % (i, route.split("/")[1].split(".")[0]), "w"
        )
        for line in script:
            fw.write(line)
