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
.. currentmodule:: qiskit.pulse

===========================
Pulse (:mod:`qiskit.pulse`)
===========================

Qiskit-Pulse is a pulse-level quantum programming kit. This lower level of programming offers the
user more control than programming with :py:class:`~qiskit.circuit.QuantumCircuit` s.

Extracting the greatest performance from quantum hardware requires real-time pulse-level
instructions. Pulse answers that need: it enables the quantum physicist *user* to specify the
exact time dynamics of an experiment. It is especially powerful for error mitigation techniques.

The input is given as arbitrary, time-ordered signals (see: :ref:`pulse-commands`) scheduled in
parallel over multiple virtual hardware or simulator resources (see: :ref:`pulse-channels`). The
system also allows the user to recover the time dynamics of the measured output.

This is sufficient to allow the quantum physicist to explore and correct for noise in a quantum
system.

.. _pulse-commands:

Commands (:mod:`~qiskit.pulse.commands`)
========================================

.. autosummary::
   :toctree: ../stubs/

   SamplePulse
   Delay
   FrameChange
   Gaussian
   GaussianSquare
   Drag
   ConstantPulse
   Acquire
   Snapshot

.. _pulse-channels:

Channels (:mod:`~qiskit.pulse.channels`)
========================================

Pulse is meant to be agnostic to the underlying hardware implementation, while still allowing
low-level control. Therefore, our signal channels are  *virtual* hardware channels. The backend
which executes our programs is responsible for mapping these virtual channels to the proper
physical channel within the quantum control hardware.

Channels are characterized by their type and their index. See each channel type below to learn more.

.. autosummary::
   :toctree: ../stubs/

   DriveChannel
   MeasureChannel
   AcquireChannel
   ControlChannel
   RegisterSlot
   MemorySlot

Schedules
=========

Schedules are Pulse programs. They describe instruction sequences for the control hardware.
An :class:`~qiskit.pulse.Instruction` is a :py:class:`~qiskit.pulse.commands.Command` which has
been assigned to its :class:`~qiskit.pulse.channels.Channel` (s).

.. autosummary::
   :toctree: ../stubs/

   Schedule
   Instruction

.. autosummary::
   :toctree: ../stubs/

   qiskit.pulse.commands
   qiskit.pulse.channels

Configuration
=============

.. autosummary::
   :toctree: ../stubs/

   InstructionScheduleMap

Rescheduling Utilities
======================

These utilities return modified :class:`~qiskit.pulse.Schedule` s.

.. autosummary::
   :toctree: ../stubs/

   ~reschedule.align_measures
   ~reschedule.add_implicit_acquires
   ~reschedule.pad

Pulse Library
=============

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.pulse.pulse_lib.discrete

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   PulseError

"""

from .channels import (DriveChannel, MeasureChannel, AcquireChannel,
                       ControlChannel, RegisterSlot, MemorySlot)
from .cmd_def import CmdDef
from .commands import (Acquire, AcquireInstruction, FrameChange,
                       PersistentValue, SamplePulse, Kernel,
                       Discriminator, Delay, ParametricPulse,
                       ParametricInstruction, Gaussian,
                       GaussianSquare, Drag, ConstantPulse, functional_pulse)
from .configuration import LoConfig, LoRange
from .exceptions import PulseError
from .instruction_schedule_map import InstructionScheduleMap
from .instructions import Instruction, Snapshot
from .interfaces import ScheduleComponent
from .schedule import Schedule
