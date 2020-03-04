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

# pylint: disable=cyclic-import

"""
===========================================
Circuit Scheduler (:mod:`qiskit.scheduler`)
===========================================

.. currentmodule:: qiskit.scheduler

A scheduler compiles a circuit program to a pulse program.

.. autosummary::
   :toctree: ../stubs/

   schedule_circuit
   ScheduleConfig

Scheduling utility functions

.. autosummary::
   :toctree: ../stubs/

   qiskit.scheduler.utils

.. automodule:: qiskit.scheduler.methods
"""

from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.schedule_circuit import schedule_circuit
from qiskit.scheduler.utils import measure, measure_all
