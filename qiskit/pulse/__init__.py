# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===========================
Pulse (:mod:`qiskit.pulse`)
===========================

Qiskit-Pulse is a pulse-level quantum programming kit. This lower level of
programming offers the user more control than programming with
:py:class:`~qiskit.circuit.QuantumCircuit` s.

Extracting the greatest performance from quantum hardware requires real-time
pulse-level instructions. Pulse answers that need: it enables the quantum
physicist *user* to specify the exact time dynamics of an experiment.
It is especially powerful for error mitigation techniques.

The input is given as arbitrary, time-ordered signals (see: :ref:`pulse-insts`)
scheduled in parallel over multiple virtual hardware or simulator resources
(see: :ref:`pulse-channels`). The system also allows the user to recover the
time dynamics of the measured output.

This is sufficient to allow the quantum physicist to explore and correct for
noise in a quantum system.

.. _pulse-insts:

Instructions (:mod:`qiskit.pulse.instructions`)
================================================

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.pulse.instructions

   Acquire
   Call
   Delay
   Play
   SetFrequency
   ShiftFrequency
   SetPhase
   ShiftPhase
   Snapshot


Pulse Library (waveforms :mod:`qiskit.pulse.library`)
=====================================================

.. autosummary::
   :toctree: ../stubs/

   library
   library.discrete

   Waveform
   Constant
   Drag
   Gaussian
   GaussianSquare

.. _pulse-channels:

Channels (:mod:`qiskit.pulse.channels`)
========================================

Pulse is meant to be agnostic to the underlying hardware implementation, while still allowing
low-level control. Therefore, our signal channels are  *virtual* hardware channels. The backend
which executes our programs is responsible for mapping these virtual channels to the proper
physical channel within the quantum control hardware.

Channels are characterized by their type and their index. See each channel type below to learn more.

.. autosummary::
   :toctree: ../stubs/

   channels

   DriveChannel
   MeasureChannel
   AcquireChannel
   ControlChannel
   RegisterSlot
   MemorySlot


Schedules
=========

Schedules are Pulse programs. They describe instruction sequences for the control hardware.

.. autosummary::
   :toctree: ../stubs/

   Schedule
   ScheduleBlock
   Instruction


Configuration
=============

.. autosummary::
   :toctree: ../stubs/

   InstructionScheduleMap


Schedule Transforms
===================

Schedule transforms take :class:`Schedule` s as input and return modified
:class:`Schedule` s.

.. autosummary::
   :toctree: ../stubs/

   transforms.align_measures
   transforms.add_implicit_acquires
   transforms.pad

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   PulseError


Pulse Builder (:mod:`~qiskit.pulse.builder`)
===================================================

.. warning::
    The pulse builder interface is still in active development. It may have
    breaking API changes without deprecation warnings in future releases until
    otherwise indicated.

The pulse builder provides an imperative API for writing pulse programs
with less difficulty than the :class:`~qiskit.pulse.Schedule` API.
It contextually constructs a pulse schedule and then emits the schedule for
execution. For example to play a series of pulses on channels is as simple as:


.. jupyter-execute::

    from qiskit import pulse

    dc = pulse.DriveChannel
    d0, d1, d2, d3, d4 = dc(0), dc(1), dc(2), dc(3), dc(4)

    with pulse.build(name='pulse_programming_in') as pulse_prog:
        pulse.play([1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], d0)
        pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], d1)
        pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], d2)
        pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], d3)
        pulse.play([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], d4)

    pulse_prog.draw()


In the future the pulse builder will be coupled to the
:class:`~qiskit.circuit.QuantumCircuit` with an equivalent circuit builder
interface.

.. autosummary::
    :toctree: ../stubs/

    ~qiskit.pulse.builder.build


Channels
--------
Methods to return the correct channels for the respective qubit indices.

.. jupyter-execute::

    from qiskit import pulse
    from qiskit.test.mock import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as drive_sched:
        d0 = pulse.drive_channel(0)
        print(d0)


.. autosummary::
    :toctree: ../stubs/

    ~qiskit.pulse.builder.acquire_channel
    ~qiskit.pulse.builder.control_channels
    ~qiskit.pulse.builder.drive_channel
    ~qiskit.pulse.builder.measure_channel


Instructions
------------
Pulse instructions are available within the builder interface. Here's an example:

.. jupyter-execute::

    from qiskit import pulse
    from qiskit.test.mock import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as drive_sched:
        d0 = pulse.drive_channel(0)
        a0 = pulse.acquire_channel(0)

        pulse.play(pulse.library.Constant(10, 1.0), d0)
        pulse.delay(20, d0)
        pulse.shift_phase(3.14/2, d0)
        pulse.set_phase(3.14, d0)
        pulse.shift_frequency(1e7, d0)
        pulse.set_frequency(5e9, d0)

        with pulse.build() as temp_sched:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), d0)
            pulse.play(pulse.library.Gaussian(20, -1.0, 3.0), d0)

        pulse.call(temp_sched)
        pulse.acquire(30, a0, pulse.MemorySlot(0))

    drive_sched.draw()


.. autosummary::
    :toctree: ../stubs/

    ~qiskit.pulse.builder.acquire
    ~qiskit.pulse.builder.barrier
    ~qiskit.pulse.builder.call
    ~qiskit.pulse.builder.delay
    ~qiskit.pulse.builder.play
    ~qiskit.pulse.builder.set_frequency
    ~qiskit.pulse.builder.set_phase
    ~qiskit.pulse.builder.shift_frequency
    ~qiskit.pulse.builder.shift_phase
    ~qiskit.pulse.builder.snapshot


Contexts
--------
Builder aware contexts that modify the construction of a pulse program. For
example an alignment context like :func:`~qiskit.pulse.builder.align_right` may
be used to align all pulses as late as possible in a pulse program.

.. jupyter-execute::

    from qiskit import pulse

    d0 = pulse.DriveChannel(0)
    d1 = pulse.DriveChannel(1)

    with pulse.build() as pulse_prog:
        with pulse.align_right():
            # this pulse will start at t=0
            pulse.play(pulse.Constant(100, 1.0), d0)
            # this pulse will start at t=80
            pulse.play(pulse.Constant(20, 1.0), d1)

    pulse_prog.draw()

.. autosummary::
    :toctree: ../stubs/

    ~qiskit.pulse.builder.align_equispaced
    ~qiskit.pulse.builder.align_func
    ~qiskit.pulse.builder.align_left
    ~qiskit.pulse.builder.align_right
    ~qiskit.pulse.builder.align_sequential
    ~qiskit.pulse.builder.circuit_scheduler_settings
    ~qiskit.pulse.builder.frequency_offset
    ~qiskit.pulse.builder.inline
    ~qiskit.pulse.builder.pad
    ~qiskit.pulse.builder.phase_offset
    ~qiskit.pulse.builder.transpiler_settings


Macros
------
Macros help you add more complex functionality to your pulse
program.

.. jupyter-execute::

    from qiskit import pulse
    from qiskit.test.mock import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as measure_sched:
        mem_slot = pulse.measure(0)
        print(mem_slot)

.. autosummary::
    :toctree: ../stubs/

    ~qiskit.pulse.builder.measure
    ~qiskit.pulse.builder.measure_all
    ~qiskit.pulse.builder.delay_qubits


Circuit Gates
-------------
To use circuit level gates within your pulse program call a circuit
with :func:`qiskit.pulse.builder.call`.

.. warning::
    These will be removed in future versions with the release of a circuit
    builder interface in which it will be possible to calibrate a gate in
    terms of pulses and use that gate in a circuit.

.. jupyter-execute::

    import math

    from qiskit import pulse
    from qiskit.test.mock import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as u3_sched:
        pulse.u3(math.pi, 0, math.pi, 0)

.. autosummary::
    :toctree: ../stubs/

    ~qiskit.pulse.builder.cx
    ~qiskit.pulse.builder.u1
    ~qiskit.pulse.builder.u2
    ~qiskit.pulse.builder.u3
    ~qiskit.pulse.builder.x


Utilities
---------
The utility functions can be used to gather attributes about the backend and modify
how the program is built.

.. jupyter-execute::

    from qiskit import pulse

    from qiskit.test.mock import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as u3_sched:
        print('Number of qubits in backend: {}'.format(pulse.num_qubits()))

        samples = 160
        print('There are {} samples in {} seconds'.format(
            samples, pulse.samples_to_seconds(160)))

        seconds = 1e-6
        print('There are {} seconds in {} samples.'.format(
            seconds, pulse.seconds_to_samples(1e-6)))

.. autosummary::
    :toctree: ../stubs/

    ~qiskit.pulse.builder.active_backend
    ~qiskit.pulse.builder.active_transpiler_settings
    ~qiskit.pulse.builder.active_circuit_scheduler_settings
    ~qiskit.pulse.builder.num_qubits
    ~qiskit.pulse.builder.qubit_channels
    ~qiskit.pulse.builder.samples_to_seconds
    ~qiskit.pulse.builder.seconds_to_samples

"""

# Builder imports.
from qiskit.pulse.builder import (
    # Construction methods.
    active_backend,
    active_transpiler_settings,
    active_circuit_scheduler_settings,
    build,
    num_qubits,
    qubit_channels,
    samples_to_seconds,
    seconds_to_samples,
    # Instructions.
    acquire,
    barrier,
    call,
    delay,
    play,
    set_frequency,
    set_phase,
    shift_frequency,
    shift_phase,
    snapshot,
    # Channels.
    acquire_channel,
    control_channels,
    drive_channel,
    measure_channel,
    # Contexts.
    align_equispaced,
    align_func,
    align_left,
    align_right,
    align_sequential,
    circuit_scheduler_settings,
    frequency_offset,
    inline,
    pad,
    phase_offset,
    transpiler_settings,
    # Macros.
    macro,
    measure,
    measure_all,
    delay_qubits,
    # Circuit instructions.
    cx,
    u1,
    u2,
    u3,
    x,
)
from qiskit.pulse.channels import (
    AcquireChannel,
    ControlChannel,
    DriveChannel,
    MeasureChannel,
    MemorySlot,
    RegisterSlot,
)
from qiskit.pulse.configuration import (
    Discriminator,
    Kernel,
    LoConfig,
    LoRange,
)
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import (
    Acquire,
    Call,
    Delay,
    Instruction,
    Play,
    SetFrequency,
    SetPhase,
    ShiftFrequency,
    ShiftPhase,
    Snapshot,
)
from qiskit.pulse.library import (
    Constant,
    Drag,
    Gaussian,
    GaussianSquare,
    ParametricPulse,
    Waveform,
)
from qiskit.pulse.library.samplers.decorators import functional_pulse
from qiskit.pulse.schedule import Schedule, ScheduleBlock
