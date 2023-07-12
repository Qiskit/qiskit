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

"""QuantumCircuit to Pulse scheduler."""
from typing import Optional

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError

from qiskit.pulse.schedule import Schedule
from qiskit.transpiler import Target
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.methods import as_soon_as_possible, as_late_as_possible
from qiskit.utils.deprecation import deprecate_arg

def convert_to_target(func):
    @deprecate_arg(
        "schedule_config",
        deprecation_description="Using target instead of schedule_config.",
        since="0.25.0",
        pending=True,
        predicate=lambda schedule_config: schedule_config is not None,
    )
    def _wrapped(circuit: QuantumCircuit, schedule_config: ScheduleConfig, target: Target):
        if schedule_config is not None:
            target = Target(schedule_config.meas_map)
            target.update_from_instruction_schedule_map(schedule_config.inst_map)
        return func(circuit, target=target)
    return _wrapped

@convert_to_target
def schedule_circuit(
    circuit: QuantumCircuit,
    schedule_config: ScheduleConfig = None,
    target: Target = None,
    method: Optional[str] = None,
) -> Schedule:
    """
    Basic scheduling pass from a circuit to a pulse Schedule, using the backend. If no method is
    specified, then a basic, as late as possible scheduling pass is performed, i.e. pulses are
    scheduled to occur as late as possible.

    Supported methods:

        * ``'as_soon_as_possible'``: Schedule pulses greedily, as early as possible on a
          qubit resource. (alias: ``'asap'``)
        * ``'as_late_as_possible'``: Schedule pulses late-- keep qubits in the ground state when
          possible. (alias: ``'alap'``)

    Args:
        circuit: The quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.
        target: Target built from some Backend parameters.
        method: The scheduling pass method to use.

    Returns:
        Schedule corresponding to the input circuit.

    Raises:
        QiskitError: If method isn't recognized.
    """
    methods = {
        "as_soon_as_possible": as_soon_as_possible,
        "asap": as_soon_as_possible,
        "as_late_as_possible": as_late_as_possible,
        "alap": as_late_as_possible,
    }
    if method is None:
        method = "as_late_as_possible"
    try:
        return methods[method](circuit, schedule_config, target)
    except KeyError as ex:
        raise QiskitError(f"Scheduling method {method} isn't recognized.") from ex
