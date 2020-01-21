""" Implementation of Worker class """
from collections import defaultdict
from ddlsym.tasks import *
from ddlsym.stats import get_stat_collector


class Worker(object):
    def __init__(self, name, num_layers, layer_sizes, fwd_speed, bwd_speed, partition_size):
        self.name = name
        self.task_queue = None
        self._inited = False
        self._servers = None
        self._num_layers = num_layers
        self._layer_sizes = layer_sizes
        self.fwd_speed = fwd_speed
        self.bwd_speed = bwd_speed
        self.partition_size = partition_size
        self.finished_queue = []

        for layer_index, layer_size in enumerate(layer_sizes):
            assert layer_size % partition_size == 0, "Non-integer partition number for layer {}.".format(layer_index)

        self._fwd_tasks = []
        self._bwd_tasks = []

    @property
    def initialized(self):
        return self._inited

    def initialize(self, servers=None):
        self._servers = servers
        self._initialize_tasks()
        self._inited = True

    def _initialize_fw_bw_tasks(self):
        """
        Create task structures.
        """

        fwd_compute_tasks = []
        bwd_compute_tasks = []

        # Create fwd compute tasks
        for i in range(self._num_layers):
            fwd_compute_tasks.append(
                TaskComputation("{}_fw_l{}".format(self.name, i), self._layer_sizes[i], "forward", i, executor=self))
        # Register fwd compute task structure
        for i in range(self._num_layers - 1):
            fwd_compute_tasks[i].register_successors(fwd_compute_tasks[i + 1])

        # Create bwd compute tasks
        for i in range(self._num_layers):
            bwd_compute_tasks.append(
                TaskComputation("{}_bw_l{}".format(self.name, i), self._layer_sizes[i], "backward", i, executor=self))

        fwd_compute_tasks[-1].register_successors(bwd_compute_tasks[-1])
        # Register bwd compute task structure
        for i in range(1, self._num_layers):
            bwd_compute_tasks[self._num_layers - i].register_successors(bwd_compute_tasks[self._num_layers - i - 1])

        self._fwd_tasks = fwd_compute_tasks
        self._bwd_tasks = bwd_compute_tasks

        fwd_compute_tasks[0].first_round = False  # Important

        # register batch_end_callback
        bwd_compute_tasks[-1].register_callback(lambda tsk: get_stat_collector().on_batch_end(self))

        self.task_queue.append(fwd_compute_tasks[0])

    def _initialize_tasks(self):
        raise NotImplementedError

    def _process_finished_queue(self):
        for cb in self.finished_queue:
            cb()
        self.finished_queue = []

    def process(self):
        raise NotImplementedError


class PSWorker(Worker):
    def __init__(self, name, num_layers, layer_sizes, fwd_speed, bwd_speed, partition_size):
        super().__init__(name, num_layers, layer_sizes, fwd_speed, bwd_speed, partition_size)
        self.task_queue = []
        self.push_tasks = []
        self.pull_tasks = []
        self.server2task = defaultdict(list)

    def _initialize_tasks(self):
        # Initiate computation tasks
        self._initialize_fw_bw_tasks()

        num_partitions = [layer_size // self.partition_size for layer_size in self._layer_sizes]

        # Create communication tasks
        for i in range(self._num_layers):
            self.push_tasks.append([
                TaskPSPush("{}_push_l{}_p{}".format(self.name, i, p), self.partition_size, i, p, self)
                for p in range(num_partitions[i])
            ])
            self.pull_tasks.append([
                TaskPSPull("{}_pull_l{}_p{}".format(self.name, i, p), self.partition_size, i, p, self)
                for p in range(num_partitions[i])
            ])

        # Assign communication tasks to servers
        flattened_push_tasks = [push_task for layer in self.push_tasks for push_task in layer]
        flattened_pull_tasks = [pull_task for layer in self.pull_tasks for pull_task in layer]
        for task_index, push_task in enumerate(flattened_push_tasks):
            pull_task = flattened_pull_tasks[task_index]

            assigned_server_index = task_index % len(self._servers)
            assigned_server = self._servers[assigned_server_index]
            # register targets
            push_task.register_executor(assigned_server)
            pull_task.register_executor(assigned_server)
            # Populate server2task dictionary
            self.server2task[assigned_server].append((push_task, pull_task))

        # Register communication tasks dependencies
        # attach push to bwd computing tasks
        for i in range(self._num_layers):
            self._bwd_tasks[i].register_successors(self.push_tasks[i])

        # Aggregation and apply ops are to be created at the server side.

        # Attach pull to fwd computing tasks
        for i in range(self._num_layers):
            for pull_task in self.pull_tasks[i]:
                pull_task.register_successors(self._fwd_tasks[i])

    def process(self):
        """
        Worker's workflow: worker only processes computation tasks in this model. All it needs to do is to get a task
        from task queue and run.
        """
        if not self._inited:
            raise RuntimeError("Worker must be initialized before processing.")

        self._process_finished_queue()

        if self.task_queue:
            current_task = self.task_queue[0]
            assert isinstance(current_task, TaskComputation)

            if current_task.task_type == "forward":
                current_speed = self.fwd_speed
            else:
                current_speed = self.bwd_speed
            done = current_task.work(current_speed)
            if done:
                self.task_queue.pop(0)
