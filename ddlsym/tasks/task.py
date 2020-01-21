""" Task implementation in system simulator """
from ..stats import StatRecord


class Task(object):
    """
    Base class for task object
    """

    def __init__(self, name, size, layer_id, executor=None, partition_id=-1):
        self.name = name
        self.size = size
        self.layer_id = layer_id
        self.executor = executor
        self.partition_id = partition_id
        self._successors = []
        self._predecessors = set()
        self._locked_predecessors = set()
        self._remaining_work = size
        self._stat_record = StatRecord(name)
        self._callbacks = []

    @property
    def is_done(self):
        if self._remaining_work <= 0:
            return True
        else:
            return False

    @property
    def is_ready(self):
        return not self._locked_predecessors

    def register_executor(self, executor):
        self.executor = executor

    def work(self, speed):
        """
        Deducts one unit of work for this task
        :return:
        """
        if not self._stat_record.started:
            self._stat_record.start()
        self._remaining_work -= speed
        if self._remaining_work <= 0:
            self._done()
            return True
        return False

    def register_callback(self, func):
        self._callbacks.append(func)

    def register_predecessors(self, predecessor):
        # NOTE: This method should not be used except in the register_successors function
        self._predecessors.add(predecessor)
        self._locked_predecessors.add(predecessor)

    def register_successors(self, successors):
        if isinstance(successors, list):
            for successor in successors:
                self._successors.append(successor)
                successor.register_predecessors(self)
        else:
            self._successors.append(successors)
            successors.register_predecessors(self)

    def unlock(self, predecessor):
        self._locked_predecessors.remove(predecessor)
        if self.is_ready:
            self._on_ready()

    def _on_ready(self):
        raise NotImplementedError

    def _reset(self):
        self._remaining_work = self.size
        self._locked_predecessors = self._predecessors.copy()

    def _on_done_cb(self):
        self._stat_record.stop()
        for successor in self._successors:
            successor.unlock(self)
        for func in self._callbacks:
            func(self)
        self._reset()

    def _done(self):
        self.executor.finished_queue.append(self._on_done_cb)


class TaskComputation(Task):
    """
    Computation task.
    """
    def __init__(self, name, size, task_type, layer_id, executor=None, partition_id=-1):
        super().__init__(name, size, layer_id, executor, partition_id)
        self.task_type = task_type
        self._stat_record.set_type(task_type)
        self.first_round = True

    def unlock(self, predecessor):
        if self.first_round:
            self._remove_comm_dependencies()
            self.first_round = False
        super().unlock(predecessor)

    def _remove_comm_dependencies(self):
        for task in self._locked_predecessors.copy():
            if isinstance(task, TaskCommunication):
                self._locked_predecessors.remove(task)

    def _on_ready(self):
        self.executor.task_queue.append(self)


class TaskCommunication(Task):
    """⁄ !  !
    Base class for communication tasks.
    """

    def __init__(self, name, size, layer_id, partition_id, source):
        super().__init__(name, size, layer_id, None, partition_id)
        self._source = source

    def _on_ready(self):
        raise NotImplementedError


class TaskPSPush(TaskCommunication):
    """
    Push task in PS architecture.
    source: worker, target: server
    """
    def __init__(self, name, size, layer_id, partition_id, source):
        super().__init__(name, size, layer_id, partition_id, source)
        self._stat_record.set_type("push")

    def _on_ready(self):
        """
        Pushes itself to server's push queue
        :return:
        """
        self.executor.receive_queue.append(self)

    def _on_done_cb(self):
        super()._on_done_cb()
        # notify server to unlock dummy for aggregation
        self.executor.on_push_finished(self.layer_id, self.partition_id)


class TaskPSPull(TaskCommunication):
    """
    Pull task in PS architecture.
    """
    def __init__(self, name, size, layer_id, partition_id, source):
        super().__init__(name, size, layer_id, partition_id, source)
        self._stat_record.set_type("pull")

    def _on_ready(self):
        """
        Pushes itself to server's send queue
        """
        self.executor.send_queue.append(self)


class TaskDummy(Task):
    """
    Dummy task, used to control dependencies
    """

    def __init__(self, name, layer_id, partition_id=-1):
        super().__init__(name, 1, layer_id, executor=None, partition_id=partition_id)
        self._stat_record.deactivate()

    def ready(self):
        self._on_ready()

    def _on_ready(self):
        # directly call cb to avoid latency
        self._on_done_cb()

    def work(self, speed):
        raise RuntimeError("Should not call work() on TaskDummies.")
