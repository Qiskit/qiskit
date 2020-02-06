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
===============================
OpenPulse (:mod:`qiskit.pulse`)
===============================

.. currentmodule:: qiskit.pulse

Channels
========

.. autosummary::
   :toctree: ../stubs/

   PulseChannelSpec
   DriveChannel
   MeasureChannel
   AcquireChannel
   ControlChannel
   RegisterSlot
   MemorySlot

Commands
========

.. autosummary::
   :toctree: ../stubs/

   Instruction
   Acquire
   AcquireInstruction
   FrameChange
   PersistentValue
   SamplePulse
   Snapshot
   Kernel
   Discriminator
   Delay
   functional_pulse
   ParametricPulse
   ParametricInstruction
   Gaussian
   GaussianSquare
   Drag
   ConstantPulse
   functional_pulse

Schedules
=========

.. autosummary::
   :toctree: ../stubs/

   Schedule
   ScheduleComponent

Configuration
=============

.. autosummary::
   :toctree: ../stubs/

   CmdDef
   LoConfig
   LoRange

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   PulseError
"""

from .channels import (PulseChannelSpec, DriveChannel,
                       MeasureChannel, AcquireChannel,
                       ControlChannel, RegisterSlot, MemorySlot)
from .cmd_def import CmdDef
from .commands import (Instruction, Acquire, AcquireInstruction, FrameChange,
                       PersistentValue, SamplePulse, Snapshot, Kernel,
                       Discriminator, Delay, ParametricPulse,
                       ParametricInstruction, Gaussian,
                       GaussianSquare, Drag, ConstantPulse, functional_pulse)
from .configuration import LoConfig, LoRange
from .exceptions import PulseError
from .instruction_schedule_map import InstructionScheduleMap
from .interfaces import ScheduleComponent
from .schedule import Schedule
