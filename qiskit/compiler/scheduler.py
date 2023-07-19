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
Convenience entry point into pulse scheduling, requiring only a circuit and a backend. For more
control over pulse scheduling, look at `qiskit.scheduler.schedule_circuit`.
"""
import logging

from time import time
from typing import List, Optional, Union

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.measure import Measure
from qiskit.exceptions import QiskitError
from qiskit.pulse import InstructionScheduleMap, Schedule
from qiskit.providers.backend import Backend, BackendV1, BackendV2
from qiskit.providers.backend_compat import convert_to_target
from qiskit.transpiler import Target
from qiskit.scheduler.schedule_circuit import schedule_circuit
from qiskit.tools.parallel import parallel_map

logger = logging.getLogger(__name__)


def _log_schedule_time(start_time, end_time):
    log_msg = "Total Scheduling Time - %.5f (ms)" % ((end_time - start_time) * 1000)
    logger.info(log_msg)


def schedule(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Optional[Backend] = None,
    inst_map: Optional[InstructionScheduleMap] = None,
    meas_map: Optional[List[List[int]]] = None,
    dt: Optional[float] = None,
    method: Optional[Union[str, List[str]]] = None,
    target: Optional[Target] = None,
) -> Union[Schedule, List[Schedule]]:
    """
    Schedule a circuit to a pulse ``Schedule``, using the backend, according to any specified
    methods. Supported methods are documented in :py:mod:`qiskit.scheduler.schedule_circuit`.

    Args:
        circuits: The quantum circuit or circuits to translate
        backend: A backend instance, which contains hardware-specific data required for scheduling
        inst_map: Mapping of circuit operations to pulse schedules. If ``None``, defaults to the
                  ``backend``\'s ``instruction_schedule_map``
        meas_map: List of sets of qubits that must be measured together. If ``None``, defaults to
                  the ``backend``\'s ``meas_map``
        dt: The output sample rate of backend control electronics. For scheduled circuits
            which contain time information, dt is required. If not provided, it will be
            obtained from the backend configuration
        method: Optionally specify a particular scheduling method
        target: The optional :class:`~.Target` representing the target backend. If ``None``,
                defaults to the ``backend``\'s ``target``, constructed from convert_to_target,
                or prepared from ``meas_map`` and ``inst_map``

    Returns:
        A pulse ``Schedule`` that implements the input circuit

    Raises:
        QiskitError: If ``inst_map`` and ``meas_map`` are not passed and ``backend`` is not passed
    """
    arg_circuits_list = isinstance(circuits, list)
    start_time = time()
    if target is None:
        if isinstance(backend, BackendV2):
            target = backend.target
            if inst_map:
                target.update_from_instruction_schedule_map(inst_map=inst_map)
        elif isinstance(backend, BackendV1):
            if hasattr(backend, "configuration"):
                target = convert_to_target(
                    configuration=backend.configuration(),
                    properties=backend.properties(),
                    defaults=backend.defaults() if hasattr(backend, "defaults") else None,
                )
                print(f"{inst_map=}")
                print(backend.defaults().instruction_schedule_map)
            else:
                target = Target(concurrent_measurements=meas_map or backend.configuration().meas_map)
                defaults=backend.defaults() if hasattr(backend, "defaults") else None
                inst_map = inst_map or defaults.instruction_schedule_map if hasattr(defaults, "instruction_schedule_map") else None
                # print(f"aaaa{inst_map=}")
                # target.add_instruction(Measure(), properties=backend.properties())
            if inst_map:
                target.update_from_instruction_schedule_map(inst_map=inst_map)
        else:
            if inst_map:
                target = Target(concurrent_measurements=meas_map)
                target.update_from_instruction_schedule_map(inst_map=inst_map)
            else:
                raise QiskitError(
                    "Must specify either target, backend, or inst_map for scheduling passes."
                )
    print(target.instruction_schedule_map())
    circuits = circuits if isinstance(circuits, list) else [circuits]
    schedules = parallel_map(schedule_circuit, circuits, (None, target, method))
    end_time = time()
    _log_schedule_time(start_time, end_time)
    if arg_circuits_list:
        return schedules
    else:
        return schedules[0]
