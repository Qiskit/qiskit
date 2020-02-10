import contextvars
from contextlib import contextmanager

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


def measure(q: int):
    cim = backend_ctx.get().circuit_instruction_map
    schedule = schedule_ctx.get()
    schedule._append(cim.get('measure', q))


def u1(q: int, P0):
    cim = backend_ctx.get().circuit_instruction_map
    schedule = schedule_ctx.get()
    schedule._append(cim.get('u1', q, P0=P0))


def u2(q: int, P0, P1):
    cim = backend_ctx.get().circuit_instruction_map
    schedule = schedule_ctx.get()
    schedule._append(cim.get('u2', q, P0=P0, P1=P1))


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
from .schedule import Schedule

def context_test(provider):
    backend = provider.get_backend('ibmq_armonk')

    schedule = Schedule()
    with build(backend, schedule):
        u2(0, 0, pi/2)
        u2(0, 0, pi)
        measure(0)

    return schedule
