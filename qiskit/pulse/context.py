from contextlib import contextmanager
import contextvars

from .channels import Channel, DriveChannel, ControlChannel, MeasureChannel, AcquireChannel
from .commands.delay import Delay
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


def measure(qubit: int):
    ism = backend_ctx.get().instruction_schedule_map
    schedule = schedule_ctx.get()
    schedule._append(ism.get('measure', qubit))


def u1(qubit: int, P0):
    ism = backend_ctx.get().instruction_schedule_map
    schedule = schedule_ctx.get()
    schedule._append(ism.get('u1', qubit, P0=P0))


def u2(qubit: int, P0, P1):
    ism = backend_ctx.get().instruction_schedule_map
    schedule = schedule_ctx.get()
    schedule._append(ism.get('u2', qubit, P0=P0, P1=P1))


def delay(qubit: int, duration: int):
    schedule = schedule_ctx.get()
    schedule._append(Delay(duration).to_instruction(DriveChannel(qubit)))
    schedule._append(Delay(duration).to_instruction(ControlChannel(qubit)))
    schedule._append(Delay(duration).to_instruction(MeasureChannel(qubit)))
    schedule._append(Delay(duration).to_instruction(AcquireChannel(qubit)))


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

def context_test(provider):
    backend = provider.get_backend('ibmq_armonk')

    schedule = Schedule()
    with build(backend, schedule):
        u2(0, 0, pi/2)
        delay(0, 100)
        u2(0, 0, pi)
        measure(0)

    return schedule
