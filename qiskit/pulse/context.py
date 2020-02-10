import contextvars
from contextlib import contextmanager

backend_ctx = contextvars.ContextVar("backend")
schedule_ctx = contextvars.ContextVar("schedule")

def rx90(q: int):
    backend = backend_ctx.get()
    schedule = schedule_ctx.get()
    schedule.append(('rx90', q))


def rx180(q: int):
    backend = backend_ctx.get()
    schedule = schedule_ctx.get()
    schedule.append(('rx180', q))

@contextmanager
def build(backend, schedule):
    token1 = backend_ctx.set([])
    token2 = schedule_ctx.set(schedule)
    try:
        yield
    finally:
        backend_ctx.reset(token1)
        schedule_ctx.reset(token2)

backend = object()
schedule = []
with build(backend, schedule):
    rx90(0)
    rx90(1)
    rx180(0)
