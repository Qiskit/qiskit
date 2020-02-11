from contextlib import contextmanager
import contextvars

from . import alignment
from .channels import Channel, DriveChannel, ControlChannel, MeasureChannel, AcquireChannel
from .commands.delay import Delay
from .commands.frame_change import FrameChange
from .commands.sample_pulse import SamplePulse
from .reschedule import pad
from .schedule import Schedule

backend_ctx = contextvars.ContextVar("backend")
schedule_ctx = contextvars.ContextVar("schedule")
instruction_list_ctx = contextvars.ContextVar("instruction_list")

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
    token3 = instruction_list_ctx.set([])
    try:
        yield
    finally:
        schedule.append(alignment.align_left(*instruction_list_ctx.get()), mutate=True)
        backend_ctx.reset(token1)
        schedule_ctx.reset(token2)
        instruction_list_ctx.reset(token3)


@contextmanager
def left_barrier():
    # clear the instruction list in this context
    token = instruction_list_ctx.set([])
    try:
        yield
    finally:
        aligned_schedule = alignment.left_barrier(*instruction_list_ctx.get())
        # restore the containing context instruction list
        instruction_list_ctx.reset(token)
        # add our aligned schedule to the outer context instruction list
        instruction_list = instruction_list_ctx.get()
        instruction_list.append(aligned_schedule)


@contextmanager
def right_barrier():
    # clear the instruction list in this context
    token = instruction_list_ctx.set([])
    try:
        yield
    finally:
        aligned_schedule = alignment.right_barrier(*instruction_list_ctx.get())
        # restore the containing context instruction list
        instruction_list_ctx.reset(token)
        # add our aligned schedule to the outer context instruction list
        instruction_list = instruction_list_ctx.get()
        instruction_list.append(aligned_schedule)


@contextmanager
def sequence():
    # clear the instruction list in this context
    token = instruction_list_ctx.set([])
    try:
        yield
    finally:
        aligned_schedule = alignment.align_in_sequence(*instruction_list_ctx.get())
        # restore the containing context instruction list
        instruction_list_ctx.reset(token)
        # add our aligned schedule to the outer context instruction list
        instruction_list = instruction_list_ctx.get()
        instruction_list.append(aligned_schedule)


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
    instruction_list = instruction_list_ctx.get()
    instruction_list.append(ism.get('measure', qubit))


def u1(qubit: int, p0):
    ism = backend_ctx.get().instruction_schedule_map
    instruction_list = instruction_list_ctx.get()
    instruction_list.append(ism.get('u1', qubit, P0=p0))


def u2(qubit: int, p0, p1):
    ism = backend_ctx.get().instruction_schedule_map
    instruction_list = instruction_list_ctx.get()
    instruction_list.append(ism.get('u2', qubit, P0=p0, P1=p1))


def u3(qubit: int, p0, p1, p2):
    ism = backend_ctx.get().instruction_schedule_map
    instruction_list = instruction_list_ctx.get()
    instruction_list.append(ism.get('u3', qubit, P0=p0, P1=p1, P2=p2))


def cx(control: int, target: int):
    ism = backend_ctx.get().instruction_schedule_map
    instruction_list = instruction_list_ctx.get()
    instruction_list.append(ism.get('cx', (control, target)))


def delay(qubit: int, duration: int):
    instruction_list = instruction_list_ctx.get()
    for ch in qubit_channels(qubit):
        instruction_list.append(Delay(duration)(ch))


def play(ch: Channel, pulse: SamplePulse):
    instruction_list = instruction_list_ctx.get()
    instruction_list.append(pulse(ch))


def shift_phase(ch: Channel, phase: float):
    instruction_list = instruction_list_ctx.get()
    instruction_list.append(FrameChange(phase)(ch))


# testing (to be moved to another module)

from math import pi
from .pulse_lib.discrete import gaussian
from qiskit.test.mock import FakeAlmaden

def context_test():
    backend = FakeAlmaden()

    schedule = Schedule()
    with build(backend, schedule):
        u2(0, 0, pi/2)
        delay(0, 1000)
        u2(0, 0, pi)
        with left_barrier():
            play(DriveChannel(0), gaussian(500, 0.1, 125))
            shift_phase(DriveChannel(0), pi/2)
            play(DriveChannel(0), gaussian(500, 0.1, 125))
            u2(1, 0, pi/2)
        with sequence():
            u2(0, 0, pi/2)
            u2(1, 0, pi/2)
            u2(0, 0, pi/2)
        # measure(0)

    return schedule
