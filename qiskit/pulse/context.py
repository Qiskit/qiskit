from contextlib import contextmanager
import contextvars

from .channels import Channel, DriveChannel, ControlChannel, MeasureChannel, AcquireChannel
from .commands.delay import Delay
from .commands.frame_change import FrameChange
from .commands.sample_pulse import SamplePulse
from .reschedule import pad
from .schedule import Schedule

backend_ctx = contextvars.ContextVar("backend")
schedule_ctx = contextvars.ContextVar("schedule")

# def rx90(q: int):
#     backend_defaults = backend_ctx.get()
#     schedule = schedule_ctx.get()
#     schedule.append(('rx90', q))
#
#
# def rx180(q: int):
#     backend_defaults = backend_ctx.get()
#     schedule = schedule_ctx.get()
#     schedule.append(('rx180', q))

def qubit_channels(qubit: int):
    """
    Returns the 'typical' set of channels associated with a qubit.
    """
    return [DriveChannel(qubit), ControlChannel(qubit), MeasureChannel(qubit),
            AcquireChannel(qubit)]


def measure(qubit: int):
    ism = backend_ctx.get().instruction_schedule_map
    schedule = schedule_ctx.get()
    schedule.append(ism.get('measure', qubit), mutate=True)
    pad(schedule, channels=qubit_channels(qubit), mutate=True)


def u1(qubit: int, p0):
    ism = backend_ctx.get().instruction_schedule_map
    schedule = schedule_ctx.get()
    schedule.append(ism.get('u1', qubit, P0=p0), mutate=True)
    pad(schedule, channels=qubit_channels(qubit), mutate=True)


def u2(qubit: int, p0, p1):
    ism = backend_ctx.get().instruction_schedule_map
    schedule = schedule_ctx.get()
    schedule.append(ism.get('u2', qubit, P0=p0, P1=p1), mutate=True)
    pad(schedule, channels=qubit_channels(qubit), mutate=True)


def delay(qubit: int, duration: int):
    schedule = schedule_ctx.get()
    for ch in qubit_channels(qubit):
        schedule.append(Delay(duration)(ch), mutate=True)


def play(ch: Channel, pulse: SamplePulse):
    schedule = schedule_ctx.get()
    schedule.append(pulse(ch), mutate=True)
    # pad(schedule, channels=qubit_channels(qubit), mutate=True)


def shift_phase(ch: Channel, phase: float):
    schedule = schedule_ctx.get()
    schedule.append(FrameChange(phase)(ch), mutate=True)


@contextmanager
def build(backend, schedule):
    """
    A context manager for the pulse DSL.

    Args:
        backend: a qiskit backend
        schedule: a *mutable* pulse Schedule
    """
    token1 = backend_ctx.set(backend.defaults())
    token2 = schedule_ctx.set(schedule)
    try:
        yield
    finally:
        backend_ctx.reset(token1)
        schedule_ctx.reset(token2)

# testing (to be moved to another module)

from math import pi
from .pulse_lib.discrete import gaussian

def context_test(provider):
    backend = provider.get_backend('ibmq_armonk')

    schedule = Schedule()
    with build(backend, schedule):
        u2(0, 0, pi/2)
        delay(0, 1000)
        u2(0, 0, pi)
        play(DriveChannel(0), gaussian(1000, 1.0, 250))
        shift_phase(DriveChannel(0), pi/2)
        play(DriveChannel(0), gaussian(1000, 1.0, 250))
        measure(0)

    return schedule
