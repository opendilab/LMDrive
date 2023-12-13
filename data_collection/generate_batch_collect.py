import os

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


routes_list = []
for route in routes:
    routes_list.append(route.split("/")[1].split(".")[0])

if not os.path.exists("batch_run"):
    os.mkdir("batch_run")

for route in routes_list:
    fw = open("batch_run/run_route_%s.sh" % route, "w")
    for i in range(4):
        fw.write("bash data_collection/bashs/sub-%d/%s.sh & \n" % (i, route))
