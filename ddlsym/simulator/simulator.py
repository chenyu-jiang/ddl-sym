import warnings
from ddlsym.workers import Worker
from ddlsym.servers import Server
from ddlsym.simulator.utils import _type_error
from ddlsym.stats import get_stat_collector


class SimulatorBase(object):
    """
    Base class for ddl simulators. It maintains the main loop for the simulation. Time in the simulator is measured in
    ticks, which increments one in each loop.
    """

    def __init__(self, duration=10000, timeline_path="./simulator_timeline.html"):
        """
        Create a simulator object.
        :param duration: duration of the simulation (in ticks).
        """
        self._duration = duration
        self._time = 0
        self._workers = []
        self._breakpoints = set()
        self.timeline_path = timeline_path
        self.finished = False
        self.initialized = False
        self.registered = False

    @property
    def now(self):
        return self._time

    @property
    def workers(self):
        return self._workers

    @property
    def breakpoints(self):
        return self._breakpoints

    def start(self, num_batches=None, duration=100000):
        """
        Start the simulation
        """

        if not self.registered:
            raise RuntimeError("Simulator must be registered before used.")
        if not self.initialized:
            raise RuntimeError("Simulator must be initialized before used.")
        if num_batches:
            self.stop_on_batch(num_batches)
            self._duration = float("inf")
        else:
            self._duration = duration
        self._run_loop()

    def pause_on(self, tick):
        """
        Set a breakpoint in specific tick of the simulation.
        :param tick: where the breakpoint locates
        """
        if not isinstance(tick, int):
            raise _type_error("integer", tick)

        if tick <= self._time:
            warnings.warn("Pause tick already passed, this breakpoint will be ignored.", RuntimeWarning)
            return

        self.breakpoints.add(tick)

    def resume(self):
        """
        Resume a simulation
        """
        if self.finished:
            raise RuntimeError("Cannot restart a finished simulation.")

        self._run_loop()

    def stop(self):
        """
        Stop a simulation. This could be helpful if the user runs the program in an interactive mode.
        """
        self.finished = True
        self._on_simulation_finish()

    def stop_on_batch(self, batch):
        get_stat_collector().register_callback_on_batch(batch, self.stop)

    def register_worker(self, worker):
        if isinstance(worker, Worker):
            self._workers.append(worker)
        else:
            raise _type_error("Worker instance", worker)

    def initialize(self):
        raise NotImplementedError

    def _run_loop(self):
        while self._time < self._duration:
            self._proceed_simulation_step()
            self._time += 1

            # check pausing conditions
            if self._time in self.breakpoints:
                self.breakpoints.remove(self._time)
                return

            # check stopping conditions
            if self.finished:
                return

        # exits the loop normally
        self.finished = True
        self._on_simulation_finish()

    def _proceed_simulation_step(self):
        raise NotImplementedError

    def _on_simulation_finish(self):
        """
        Default implementation, does nothing
        :return:
        """
        pass


class SimulatorPS(SimulatorBase):
    def __init__(self, duration=10000, timeline_path="./simulator_timeline.html"):
        super().__init__(duration, timeline_path)
        self._servers = []

    @property
    def servers(self):
        return self._servers

    def register_server(self, server):
        if isinstance(server, Server):
            self._servers.append(server)
        else:
            raise _type_error("Server instance", server)

    def initialize(self):
        for worker in self._workers:
            worker.initialize(self._servers)
        for server in self._servers:
            server.initialize(self._workers)
        self.initialized = True

    def _proceed_simulation_step(self):
        for worker in self._workers:
            worker.process()
        for server in self._servers:
            server.process()

    def _on_simulation_finish(self):
        get_stat_collector().dump(self.timeline_path)
