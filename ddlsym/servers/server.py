""" Implementation of server class """
from ddlsym.tasks import *


class Server(object):
    """
    Server implementation with no explicit receive and send control
    """
    def __init__(self, name, agg_speed, apply_speed, bandwidth):
        self.name = name
        self.task_queue = []
        self.send_queue = []
        self.receive_queue = []
        self._inited = False
        self._workers = None
        self.num_workers = 0
        self.bandwidth = bandwidth
        self.agg_speed = agg_speed
        self.apply_speed = apply_speed
        self.num_partitions = []
        self.finished_queue = []
        self._aggregation_tasks = []
        self._application_tasks = []
        self._dummy_tasks = []
        self._dummy_counter = []

    def initialize(self, workers):
        self._workers = workers
        self.num_workers = len(workers)
        self._initialize_tasks()

    def _initialize_tasks(self):
        num_layers = len(self._workers[0].push_tasks)
        self.num_partitions = [len(layer) for layer in self._workers[0].push_tasks]
        partition_size = self._workers[0].push_tasks[0][0].size

        # Sanity check
        for worker in self._workers:
            assert worker.initialized, "Workers must be initialized before initializing servers."
            assert len(worker.push_tasks) == num_layers, "Workers have different number of layers."
            for layer_id, layer in enumerate(worker.push_tasks):
                assert len(layer) == self.num_partitions[layer_id], \
                    "Workers have different number of partitions for layer {}.".format(layer_id)

        # Create aggregation and application tasks for each layer
        for i in range(num_layers):
            self._aggregation_tasks.append([
                [TaskComputation("{}_agg_l{}_p{}_{}".format(self.name, i, p, wid), partition_size, "aggregation", i,
                                 executor=self, partition_id=p) for wid in range(self.num_workers)]
                for p in range(self.num_partitions[i])
            ])
            self._application_tasks.append([
                TaskComputation("{}_apply_l{}_p{}".format(self.name, i, p), partition_size, "application", i,
                                executor=self, partition_id=p)
                for p in range(self.num_partitions[i])
            ])
            self._dummy_tasks.append([
                [TaskDummy("{}_dummy_l{}_p{}_{}".format(self.name, i, p, wid), i, p)
                 for wid in range(self.num_workers)]
                for p in range(self.num_partitions[i])
            ])
            self._dummy_counter.append([
                0 for _ in range(self.num_partitions[i])
            ])

        # Register dummies for aggregation tasks
        for layer_id in range(num_layers):
            for partition_id in range(self.num_partitions[layer_id]):
                for wid in range(self.num_workers):
                    self._dummy_tasks[layer_id][partition_id][wid].register_successors(
                        self._aggregation_tasks[layer_id][partition_id][wid])

        # Register dependencies for apply tasks
        for layer_id, layer in enumerate(self._aggregation_tasks):
            for partition_id, partition in enumerate(layer):
                partition[-1].register_successors(self._application_tasks[layer_id][partition_id])

        # Get pull tasks from workers, register apply tasks
        for worker in self._workers:
            for _, pull_task in worker.server2task[self]:
                layer_id = pull_task.layer_id
                partition_id = pull_task.partition_id
                self._application_tasks[layer_id][partition_id].register_successors(pull_task)

    def _process_finished_queue(self):
        for cb in self.finished_queue:
            cb()
        self.finished_queue = []

    def on_push_finished(self, layer_id, partition_id):
        dummy = self._dummy_tasks[layer_id][partition_id][self._dummy_counter[layer_id][partition_id]]
        dummy.ready()
        self._dummy_counter[layer_id][partition_id] += 1
        if self._dummy_counter[layer_id][partition_id] == self.num_workers:
            self._dummy_counter[layer_id][partition_id] = 0

    def process(self):
        """
        PS workflow:
            1. Process push requests from receive queue (possibly add finished tasks to task_queue)
            2. Process task queue requests (possibly add pull to send_queue)
            3. Process send queue
        """

        self._process_finished_queue()

        if self.receive_queue:
            effective_bandwidth = self.bandwidth / len(self.receive_queue)
            for push_task in self.receive_queue:
                done = push_task.work(effective_bandwidth)
                if done:
                    self.receive_queue.remove(push_task)

        if self.task_queue:
            current_task = self.task_queue[0]
            assert isinstance(current_task, TaskComputation)
            if current_task.task_type == "aggregation":
                current_speed = self.agg_speed
            else:
                current_speed = self.apply_speed
            done = current_task.work(current_speed)
            if done:
                self.task_queue.pop(0)

        if self.send_queue:
            effective_bandwidth = self.bandwidth / len(self.send_queue)
            for pull_task in self.send_queue:
                done = pull_task.work(effective_bandwidth)
                if done:
                    self.send_queue.remove(pull_task)
