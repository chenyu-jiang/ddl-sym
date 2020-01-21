from .simulator import SimulatorBase as _SimulatorBase
from .simulator import SimulatorPS

_simulator = None


def register_simulator(simulator_instance):
    global _simulator
    _simulator = simulator_instance
    simulator_instance.registered = True


def get_time():
    return _simulator.now
