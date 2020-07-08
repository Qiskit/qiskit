# -*- coding: utf-8 -*-

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


.. _pulse-builder:


(BETA) Pulse Builder (:mod:`~qiskit.pulse.builder`)
===================================================

.. autosummary::
    :toctree: ../stubs/

    ~builder


.. _pulse-insts:

Instructions (:mod:`qiskit.pulse.instructions`)
================================================

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.pulse.instructions

   Acquire
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

   SamplePulse
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
   Instruction

Configuration
=============

.. autosummary::
   :toctree: ../stubs/

   InstructionScheduleMap

Schedule Transforms
===================

These functions take :class:`Schedule` s as input and return modified
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
from qiskit.pulse.commands import (
    AcquireInstruction,
    FrameChange,
    PersistentValue,
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
    Delay,
    Instruction,
    Play,
    SetFrequency,
    SetPhase,
    ShiftFrequency,
    ShiftPhase,
    Snapshot,
)
from qiskit.pulse.interfaces import ScheduleComponent
from qiskit.pulse.library import (
    Constant,
    ConstantPulse,
    Drag,
    Gaussian,
    GaussianSquare,
    ParametricPulse,
    SamplePulse,
)
from qiskit.pulse.library.samplers.decorators import functional_pulse
from qiskit.pulse.schedule import Schedule
