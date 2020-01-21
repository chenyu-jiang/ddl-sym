import argparse
import yaml
from ddlsym import simulator as dsim
from ddlsym import workers, servers
from ddlsym.stats import get_stat_collector


def configure_parser():
    parser = argparse.ArgumentParser(description="A python-based distributed deep learning system simulator.")
    parser.add_argument("-i", "--conf", default="./conf.yaml", help="Path to config file containing system setup.")
    parser.add_argument("-o", "--output", default="./timeline.json", help="Path to the output trace file.")
    parser.add_argument("-p", "--profile", default=False, help="Enable timeline.")
    args = parser.parse_args()
    return args.conf, args.output, args.profile


if __name__ == '__main__':
    conf_path, output_path, enable_profile = configure_parser()

    with open(conf_path, "r") as s:
        config = yaml.safe_load(s)

    # create simulator
    simulator = dsim.SimulatorPS(timeline_path=output_path)
    dsim.register_simulator(simulator)

    # create worker and server instances
    worker_instances = []
    server_instances = []
    layer_sizes = config["layer_sizes"] if isinstance(config["layer_sizes"], list) else \
        [config["layer_sizes"] for _ in range(config["num_layers"])]
    for worker_conf in config["workers"]:
        worker_instances.append(workers.PSWorker(worker_conf["name"], config["num_layers"], layer_sizes,
                                worker_conf["fwd_speed"], worker_conf["bwd_speed"], config["partition_size"]))
    for server_conf in config["servers"]:
        server_instances.append(servers.Server(server_conf["name"], server_conf["agg_speed"],
                                               server_conf["apply_speed"], server_conf["bandwidth"]))

    for worker in worker_instances:
        simulator.register_worker(worker)
    for server in server_instances:
        simulator.register_server(server)

    # initialize stat collector
    get_stat_collector().initialize(worker_instances, server_instances)
    if enable_profile or config["enable_profile"]:
        get_stat_collector().activate()
        simulator.stop_on_batch(10)

    # initialize simulator
    simulator.initialize()

    # start simulator
    if config["num_batches"]:
        simulator.start(config["num_batches"])
    elif config["duration"]:
        simulator.start(config["duration"])
    else:
        simulator.start()




