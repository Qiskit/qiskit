# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Context based pulse programming interface.

Use the context builder interface to program pulse programs with assembly-like
syntax. For example::

.. code::
    from qiskit.circuit import QuantumCircuit
    from qiskit.pulse import pulse_lib, Schedule, Gaussian, DriveChannel
    from qiskit.test.mock import FakeOpenPulse2Q

    sched = Schedule()

    # This creates a PulseProgramBuilderContext(sched, backend=backend)
    # instance internally and wraps in a
    # dsl builder context
    with builder(sched, backend=backend):
      # create a pulse
      gaussian_pulse = pulse_lib.gaussian(10, 1.0, 2)
      # create a channel type
      d0 = DriveChannel(0)
      d1 = DriveChannel(1)
      # play a pulse at time=0
      play(d0, gaussian_pulse)
      # play another pulse directly after at t=10
      play(d0, gaussian_pulse)
      # The default scheduling behavior is to schedule pulse in parallel
      # across independent resources, for example
      # play the same pulse on a different channel at t=0
      play(d1, gaussian_pulse)

      # We also provide alignment contexts
      # if channels are not supplied defaults to all channels
      # this context starts at t=10 due to earlier pulses
      with sequential():
        play(d0, gaussian_pulse)
        # play another pulse after at t=20
        play(d1, gaussian_pulse)

        # we can also layer contexts as each instruction is contained in its
        # local scheduling context (block).
        # Scheduling contexts are layered, and the output of a child context is
        # a fixed scheduled block in its parent context.
        # starts at t=20
        with parallel():
          # start at t=20
          play(d0, gaussian_pulse)
          # start at t=20
          play(d1, gaussian_pulse)
        # context ends at t=30

      # We also support different alignment contexts
      # Default is
      # with left():

      # all pulse instructions occur as late as possible
      with right_align():
        set_phase(d1, math.pi)
        # starts at t=30
        delay(d0, 100)
        # ends at t=130

        # starts at t=120
        play(d1, gaussian_pulse)
        # ends at t=130

      # acquire a qubit
      acquire(0, ClassicalRegister(0))
      # maps to
      #acquire(AcquireChannel(0), ClassicalRegister(0))

      # We will also support a variety of helper functions for common operations

      # measure all qubits
      # Note that as this DSL is pure Python
      # any Python code is accepted within contexts
      for i in range(n_qubits):
        measure(i, ClassicalRegister(i))

      # delay on a qubit
      # this requires knowledge of which channels belong to which qubits
      delay(0, 100)

      # insert a quantum circuit. This functions by behind the scenes calling
      # the scheduler on the given quantum circuit to output a new schedule
      # NOTE: assumes quantum registers correspond to physical qubit indices
      qc = QuantumCircuit(2, 2)
      qc.cx(0, 1)
      call(qc)
      # We will also support a small set of standard gates
      u3(0, 0, np.pi, 0)
      cx(0, 1)


      # It is also be possible to call a preexisting
      # schedule constructed with another
      # NOTE: once internals are fleshed out, Schedule may not be the default class
      tmp_sched = Schedule()
      tmp_sched += Play(dc0, gaussian_pulse)
      call(tmp_sched)

      # We also support:

      # frequency instructions
      set_frequency(dc0, 5.0e9)
      shift_frequency(dc0, 0.1e9)

      # phase instructions
      set_phase(dc0, math.pi)
      shift_phase(dc0, 0.1)

      # offset contexts
      with phase_offset(d0, math.pi):
        play(d0, gaussian_pulse)

      with frequency_offset(d0, 0.1e9):
        play(d0, gaussian_pulse)
"""

from contextlib import contextmanager
import contextvars

from .circuit_scheduler import measure as measure_schedule

from . import reschedule
from .channels import Channel
from .commands.delay import Delay
from .commands.frame_change import FrameChange
from .commands.sample_pulse import SamplePulse


BACKEND_CTX = contextvars.ContextVar("backend")
SCHEDULE_CTX = contextvars.ContextVar("schedule")
INSTRUCTION_LIST_CTX = contextvars.ContextVar("instruction_list")


@contextmanager
def build(backend, schedule):
    """
    A context manager for the pulse DSL.

    Args:
        backend: a qiskit backend
        schedule: a *mutable* pulse Schedule
    """
    backend_ctx_token = BACKEND_CTX.set(backend)
    schedule_ctx_token = SCHEDULE_CTX.set(schedule)
    instruction_list_ctx_token = INSTRUCTION_LIST_CTX.set([])
    try:
        yield
    finally:
        schedule.append(reschedule.left_align(*INSTRUCTION_LIST_CTX.get()),
                        mutate=True)
        BACKEND_CTX.reset(backend_ctx_token)
        SCHEDULE_CTX.reset(schedule_ctx_token)
        INSTRUCTION_LIST_CTX.reset(instruction_list_ctx_token)


@contextmanager
def left_barrier():
    # clear the instruction list in this context
    token = INSTRUCTION_LIST_CTX.set([])
    try:
        yield
    finally:
        aligned_schedule = reschedule.left_barrier(*INSTRUCTION_LIST_CTX.get())
        # restore the containing context instruction list
        INSTRUCTION_LIST_CTX.reset(token)
        # add our aligned schedule to the outer context instruction list
        instruction_list = INSTRUCTION_LIST_CTX.get()
        instruction_list.append(aligned_schedule)


@contextmanager
def right_barrier():
    # clear the instruction list in this context
    token = INSTRUCTION_LIST_CTX.set([])
    try:
        yield
    finally:
        aligned_schedule = reschedule.right_barrier(*INSTRUCTION_LIST_CTX.get())
        # restore the containing context instruction list
        INSTRUCTION_LIST_CTX.reset(token)
        # add our aligned schedule to the outer context instruction list
        instruction_list = INSTRUCTION_LIST_CTX.get()
        instruction_list.append(aligned_schedule)


@contextmanager
def left_align():
    # clear the instruction list in this context
    token = INSTRUCTION_LIST_CTX.set([])
    try:
        yield
    finally:
        aligned_schedule = reschedule.left_align(*INSTRUCTION_LIST_CTX.get())
        # restore the containing context instruction list
        INSTRUCTION_LIST_CTX.reset(token)
        # add our aligned schedule to the outer context instruction list
        instruction_list = INSTRUCTION_LIST_CTX.get()
        instruction_list.append(aligned_schedule)


@contextmanager
def right_align():
    # clear the instruction list in this context
    token = INSTRUCTION_LIST_CTX.set([])
    try:
        yield
    finally:
        aligned_schedule = reschedule.right_align(*INSTRUCTION_LIST_CTX.get())
        # restore the containing context instruction list
        INSTRUCTION_LIST_CTX.reset(token)
        # add our aligned schedule to the outer context instruction list
        instruction_list = INSTRUCTION_LIST_CTX.get()
        instruction_list.append(aligned_schedule)


@contextmanager
def sequential():
    # clear the instruction list in this context
    token = INSTRUCTION_LIST_CTX.set([])
    try:
        yield
    finally:
        aligned_schedule = reschedule.align_in_sequence(
            *INSTRUCTION_LIST_CTX.get())
        # restore the containing context instruction list
        INSTRUCTION_LIST_CTX.reset(token)
        # add our aligned schedule to the outer context instruction list
        instruction_list = INSTRUCTION_LIST_CTX.get()
        instruction_list.append(aligned_schedule)


def qubit_channels(qubit: int):
    """
    Returns the 'typical' set of channels associated with a qubit.
    """
    raise NotImplementedError('Qubit channels is not yet implemented.')


def measure(qubit: int):
    sched = measure_schedule(
        qubits=[qubit],
        inst_map=BACKEND_CTX.get().defaults().instruction_schedule_map,
        meas_map=BACKEND_CTX.get().configuration().meas_map)
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(sched)


def u1(qubit: int, p0):
    ism = BACKEND_CTX.get().defaults().instruction_schedule_map
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(ism.get('u1', qubit, P0=p0))


def u2(qubit: int, p0, p1):
    ism = BACKEND_CTX.get().defaults().instruction_schedule_map
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(ism.get('u2', qubit, P0=p0, P1=p1))


def u3(qubit: int, p0, p1, p2):
    ism = BACKEND_CTX.get().defaults().instruction_schedule_map
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(ism.get('u3', qubit, P0=p0, P1=p1, P2=p2))


def cx(control: int, target: int):
    ism = BACKEND_CTX.get().defaults().instruction_schedule_map
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(ism.get('cx', (control, target)))


def delay_qubit(qubit: int, duration: int):
    instruction_list = INSTRUCTION_LIST_CTX.get()
    for channel in qubit_channels(qubit):
        instruction_list.append(Delay(duration)(channel))


def delay(channel: Channel, duration: int):
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(Delay(duration)(Channel))


def play(channel: Channel, pulse: SamplePulse):
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(pulse(channel))


def shift_phase(channel: Channel, phase: float):
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(FrameChange(phase)(channel))


def x(qubit: int):
    ism = BACKEND_CTX.get().defaults().instruction_schedule_map
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(ism.get('x', qubit))
