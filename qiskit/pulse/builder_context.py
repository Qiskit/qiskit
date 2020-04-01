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
import contextvars
import functools
from contextlib import contextmanager
from typing import Callable, Union

from . import Pulse, PulseError, transforms
from .channels import (AcquireChannel, Channel, MemorySlot,
                       PulseChannel, RegisterSlot)
from .circuit_scheduler import measure as measure_schedule
from .configuration import Discriminator, Kernel
from .instructions import (Acquire, Delay, Instruction, Play,
                           SetFrequency, ShiftPhase, Snapshot)
from .schedule import Schedule


#: contextvars.ContextVar[BuilderContext]: current builder
BUILDER_CONTEXT = contextvars.ContextVar("backend")


class BuilderContext():
    """Builder context class."""

    def __init__(self, backend, block: Schedule = None):
        """Initialize builder context.

        TODO: This should contain a builder class rather than manipulating the
        IR directly.

        Args:
            backend (BaseBackend):
        """

        #: BaseBackend: Backend instance for context builder.
        self.backend = backend

        #: Schedule: Current current schedule of BuilderContext.
        self.block = None

        if block is None:
            block = Schedule()

        self.set_current_block(block)

        #: Set[Schedule]: Collection of all builder blocks.
        self.blocks = set()

        #: Schedule: Final Schedule program block.
        self._program = block

    def __enter__(self):
        """Enter Builder Context."""
        self._backend_ctx_token = BUILDER_CONTEXT.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit Builder Context."""
        BUILDER_CONTEXT.reset(self._backend_ctx_token)

    def set_current_block(self, block: Schedule) -> Schedule:
        """Set the current block."""
        assert isinstance(block, Schedule)
        self.block = block
        return block

    def compile(self) -> Schedule:
        """Compile final pulse schedule program."""
        return self._program


def build(backend, schedule):
    """
    A context manager for the pulse DSL.

    Args:
        backend: a qiskit backend
        schedule: a *mutable* pulse Schedule
    """
    return BuilderContext(backend, schedule)


# Transform Contexts ###########################################################
def _transform_context(transform: Callable) -> Callable:
    """A tranform context.

    Args:
        transform: Transform to decorate as context.
    """
    @functools.wraps(transform)
    def wrap(fn):
        @contextmanager
        def wrapped_transform(blocks, *args, **kwargs):
            builder = BUILDER_CONTEXT.get()
            block = builder.set_current_block(Schedule())
            try:
                yield
            finally:
                builder.set_current_block(transform(block, *args, **kwargs))

        return wrapped_transform

    return wrap


@_transform_context(transforms.left_barrier)
def left_barrier():
    """Left barrier transform builder context."""


@_transform_context(transforms.right_barrier)
def right_barrier():
    """Right barrier transform builder context."""


@_transform_context(transforms.left_align)
def left_align():
    """Left align transform builder context."""


@_transform_context(transforms.right_align)
def right_align():
    """Right align transform builder context."""


@_transform_context(transforms.sequential)
def sequential():
    """Sequential transform builder context."""


# Base Instructions ############################################################
def _instruction(instruction_fn):
    """Decorator that wraps a function that generates instructions and appends
    to current block.
    """
    @functools.wraps(instruction_fn)
    def instruction_wrapper(*args, **kwargs) -> Instruction:
        current_block = BUILDER_CONTEXT.get().current_block
        instruction = instruction_fn(*args, **kwargs)
        current_block.append(instruction)
        return instruction
    return instruction_wrapper


@_instruction
def delay(channel: Channel, duration: int) -> Delay:
    """Delay on a ``channel`` for a ``duration``."""
    return Delay(duration, channel)


@_instruction
def play(channel: PulseChannel, pulse: Pulse) -> Play:
    """Play a ``pulse`` on a ``channel``."""
    return Play(pulse, channel)


@_instruction
def acquire(channel: Union[AcquireChannel, int],
            register: Union[RegisterSlot, MemorySlot],
            duration: int,
            **metadata: Union[Kernel, Discriminator]
            ) -> Acquire:
    """Acquire for a ``duration`` on a ``channel`` and store the result in a ``register``."""
    if isinstance(register, MemorySlot):
        return Acquire(duration, channel, mem_slot=register, **metadata)
    elif isinstance(register, RegisterSlot):
        return Acquire(duration, channel, reg_slot=register, **metadata)
    raise PulseError(
        'Register of type: "{}" is not supported'.format(type(register)))


@_instruction
def set_frequency(channel: PulseChannel, frequency: float):
    """Set the ``frequency`` of a pulse ``channel``."""
    return SetFrequency(frequency, channel)


@_instruction
def shift_frequency(channel: PulseChannel, frequency: float):
    """Shift the ``frequency`` of a pulse ``channel``."""
    raise NotImplementedError()


@_instruction
def set_phase(channel: PulseChannel, phase: float):
    """Set the ``phase`` of a pulse ``channel``."""
    raise NotImplementedError()


@_instruction
def shift_phase(channel: PulseChannel, phase: float):
    """Shift the ``phase`` of a pulse ``channel``."""
    return ShiftPhase(phase, channel)


# Macro Instructions ###########################################################
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


def x(qubit: int):
    ism = BACKEND_CTX.get().defaults().instruction_schedule_map
    instruction_list = INSTRUCTION_LIST_CTX.get()
    instruction_list.append(ism.get('x', qubit))
