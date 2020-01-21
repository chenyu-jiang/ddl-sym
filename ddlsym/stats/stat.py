import json
import ddlsym.simulator


def id_gen():
    _id = 0
    while True:
        yield _id
        _id += 1


_id_gen = id_gen()


class SimulatorStatCollector(object):
    def __init__(self):
        self.log = []
        self.meta_events = []
        self.name2pid = {}
        self.type2cat = {"aggregation": "computation", "application": "computation",
                         "forward": "computation", "backward": "computation",
                         "push": "communication", "pull": "communication"}
        self.cat2tid = {"computation": 1, "communication": 2}
        self.active = False
        self.num_batches = {}
        self.callbacks = {}

    def activate(self):
        self.active = True

    def record(self, stat_record):
        if self.active:
            for p_name, pid in self.name2pid.items():
                if stat_record["name"].startswith(p_name):
                    stat_record["pid"] = pid
            stat_record["tid"] = self.cat2tid[self.type2cat[stat_record["cat"]]]
            if self.type2cat[stat_record["cat"]] == "communication":
                new_id = next(_id_gen)
                stat_record_start = stat_record.copy()
                stat_record_start["ph"] = "b"
                stat_record_start["id"] = new_id
                stat_record_start.pop("dur")
                stat_record_end = stat_record.copy()
                stat_record_end["ph"] = "e"
                stat_record_end.pop("dur")
                stat_record_end["ts"] = stat_record["ts"] + stat_record["dur"]
                stat_record_end["id"] = new_id
                self.log.append(stat_record_start)
                self.log.append(stat_record_end)
            else:
                self.log.append(stat_record)

    def register_callback_on_batch(self, batch_num, cb):
        self.callbacks[batch_num] = cb

    def initialize(self, workers, servers=None):
        for worker_index, worker in enumerate(workers):
            pid = 1000 + worker_index
            self.meta_events.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": worker.name}})
            for cat, tid in self.cat2tid.items():
                self.meta_events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                                         "args": {"name": cat}})
            self.name2pid[worker.name] = pid

            # initialize batch count
            self.num_batches[worker] = (0, 0)
        if servers:
            for server_index, server in enumerate(servers):
                pid = 1000 + len(workers) + server_index
                self.meta_events.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": server.name}})
                self.name2pid[server.name] = pid
                for cat, tid in self.cat2tid.items():
                    self.meta_events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                                             "args": {"name": cat}})
        self.log += self.meta_events

    def dump(self, path):
        if self.active:
            json_obj = {"traceEvents": self.log, "displayTimeUnit": "ms"}
            with open(path, "w") as f:
                json.dump(json_obj, f)
        print("")
        print("=" * 10 + " Profile Results " + "=" * 10)
        print("\nTotal time: {} units.\n".format(ddlsym.simulator.get_time()))
        for worker, (batch_count, last_finish_time) in self.num_batches.items():
            print("{}: executed {} batches, speed: {} units / batch.".format(worker.name, batch_count,
                                                                             last_finish_time/batch_count))

    def on_batch_end(self, worker):
        (num_batch, time) = self.num_batches[worker]
        self.num_batches[worker] = (num_batch+1, ddlsym.simulator.get_time())
        min_worker_batch_nums = float('inf')
        for worker, (batch_num_for_worker, _) in self.num_batches.items():
            min_worker_batch_nums = min(batch_num_for_worker, min_worker_batch_nums)
        for batch_num, cb in self.callbacks.copy().items():
            if batch_num <= min_worker_batch_nums:
                cb()
                self.callbacks.pop(batch_num)


class StatRecord(object):
    def __init__(self, name, op_type=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.op_type = op_type
        self.started = False
        self.active = True

    def deactivate(self):
        if self.started:
            raise RuntimeError("Cannot deactivate a running record.")
        self.active = False

    def activate(self):
        self.active = True

    def set_type(self, op_type):
        self.op_type = op_type

    def start(self):
        if self.active:
            self.start_time = ddlsym.simulator.get_time()
            self.started = True
            # print("[{}] Started at {}.".format(self.name, self.start_time))

    def stop(self):
        if self.started and self.active:
            self.end_time = ddlsym.simulator.get_time()
            self.started = False
            self._send()
            # print("[{}] Stopped at {}.".format(self.name, self.end_time))

    def to_dict(self):
        return dict(Task=self.name, Start=self.start_time, Finish=self.end_time, Type=self.op_type)

    def format(self):
        return {"name": self.name, "cat": self.op_type, "ph": "X", "ts": self.start_time,
                "dur": self.end_time - self.start_time, "pid": 555, "tid": 12}

    def _send(self):
        get_stat_collector().record(self.format())


_simulator_stat = SimulatorStatCollector()


def get_stat_collector():
    return _simulator_stat
