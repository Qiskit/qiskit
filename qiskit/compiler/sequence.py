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

"""
Mapping a scheduled ``QuantumCircuit`` to a pulse ``Schedule``.
"""
from typing import List, Optional, Union

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.pulse import InstructionScheduleMap, Schedule

from qiskit.compiler.schedule import schedule


def sequence(scheduled_circuits: Union[QuantumCircuit, List[QuantumCircuit]],
             backend: Optional[BaseBackend] = None,
             inst_map: Optional[InstructionScheduleMap] = None,
             meas_map: Optional[List[List[int]]] = None) -> Union[Schedule, List[Schedule]]:
    """
    Schedule a scheduled circuit to a pulse ``Schedule``, using the backend.

    Args:
        scheduled_circuits: The scheduled quantum circuit or circuits to translate
        backend: A backend instance, which contains hardware-specific data required for scheduling
        inst_map: Mapping of circuit operations to pulse schedules. If ``None``, defaults to the
                  ``backend``\'s ``instruction_schedule_map``
        meas_map: List of sets of qubits that must be measured together. If ``None``, defaults to
                  the ``backend``\'s ``meas_map``

    Returns:
        A pulse ``Schedule`` that implements the input circuit
    """
    return schedule(scheduled_circuits, backend=backend, inst_map=inst_map,
                    meas_map=meas_map, method='sequence', dt=1)
