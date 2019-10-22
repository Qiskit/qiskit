# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===================================
Scheduler (:mod:`qiskit.scheduler`)
===================================

Module for scheduling pulse `Schedule`\'s from `QuantumCircuit`\'s.

.. currentmodule:: qiskit.scheduler

.. autosummary::
   :toctree: ../stubs/

   ScheduleConfig
   schedule_circuit
   format_meas_map

Scheduling policies
===================

.. autosummary::
   :toctree: ../stubs/

   as_soon_as_possible
   as_late_as_possible
   translate_gates_to_pulse_defs

"""

from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.schedule_circuit import schedule_circuit
from qiskit.scheduler.utils import format_meas_map
from qiskit.scheduler.methods.basic import (as_late_as_possible,
                                            as_soon_as_possible,
                                            translate_gates_to_pulse_defs)
