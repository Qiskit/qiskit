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
===========================================
Circuit Scheduler (:mod:`qiskit.pulse.circuit_scheduler`)
===========================================

.. currentmodule:: qiskit.pulse.circuit_scheduler

A circuit scheduler compiles a circuit program to a pulse program.

.. autosummary::
   :toctree: ../stubs/

   schedule_circuit
   ScheduleConfig

.. automodule:: qiskit.pulse.circuit_scheduler.methods
"""

from .config import ScheduleConfig
from .schedule_circuit import schedule_circuit
from qiskit.pulse.macros import measure, measure_all
