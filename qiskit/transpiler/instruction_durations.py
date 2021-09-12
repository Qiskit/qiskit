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

"""Durations of instructions, one of transpiler configurations."""
import warnings
from typing import Optional, List, Tuple, Union, Iterable, Set

from qiskit.circuit import Barrier, Delay
from qiskit.circuit import Instruction, Qubit, ParameterExpression
from qiskit.circuit.duration import duration_in_dt
from qiskit.providers import BaseBackend
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.units import apply_prefix


class InstructionDurations:
    """Helper class to provide durations of instructions for scheduling.

    It stores durations (gate lengths) and dt to be used at the scheduling stage of transpiling.
    It can be constructed from ``backend`` or ``instruction_durations``,
    which is an argument of :func:`transpile`.
    """

    def __init__(
        self, instruction_durations: Optional["InstructionDurationsType"] = None, dt: float = None
    ):
        self.duration_by_name = {}
        self.duration_by_name_params = {}
        self.duration_by_name_qubits = {}
        self.duration_by_name_params_qubits = {}
        self.dt = dt
        if instruction_durations:
            self.update(instruction_durations)

    def __str__(self):
        """Return a string representation of all stored durations."""
        string = ""
        for k, v in self.duration_by_name.items():
            string += k
            string += ": "
            string += str(v[0]) + " " + v[1]
            string += "\n"
        for k, v in self.duration_by_name_params.items():
            string += k[0] + str(k[1])
            string += ": "
            string += str(v[0]) + " " + v[1]
            string += "\n"
        for k, v in self.duration_by_name_qubits.items():
            string += k[0] + str(k[1])
            string += ": "
            string += str(v[0]) + " " + v[1]
            string += "\n"
        for k, v in self.duration_by_name_params_qubits.items():
            string += k[0] + str(k[1]) + str(k[2])
            string += ": "
            string += str(v[0]) + " " + v[1]
            string += "\n"
        return string

    @classmethod
    def from_backend(cls, backend: BaseBackend):
        """Construct an :class:`InstructionDurations` object from the backend.

        Args:
            backend: backend from which durations (gate lengths) and dt are extracted.

        Returns:
            InstructionDurations: The InstructionDurations constructed from backend.

        Raises:
            TranspilerError: If dt and dtm is different in the backend.
        """
        # FIXME: no field currently provisioned for backends to report durations
        # dependent on gate parameter. Assume None for all params.

        # backend reports all gate_length in seconds
        instruction_durations = []
        for gate, insts in backend.properties()._gates.items():
            for qubits, props in insts.items():
                if "gate_length" in props:
                    gate_length = props["gate_length"][0]  # Throw away datetime at index 1
                    instruction_durations.append((gate, None, qubits, gate_length, "s"))
        for q, props in backend.properties()._qubits.items():
            if "readout_length" in props:
                readout_length = props["readout_length"][0]  # Throw away datetime at index 1
                instruction_durations.append(("measure", None, [q], readout_length, "s"))

        try:
            dt = backend.configuration().dt
        except AttributeError:
            dt = None

        return InstructionDurations(instruction_durations, dt=dt)

    def update(self, inst_durations: Optional["InstructionDurationsType"], dt: float = None):
        """Update self with inst_durations (inst_durations overwrite self).

        Args:
            inst_durations: Instruction durations to be merged into self (overwriting self).
            dt: Sampling duration in seconds of the target backend.

        Returns:
            InstructionDurations: The updated InstructionDurations.

        Raises:
            TranspilerError: If the format of instruction_durations is invalid.
        """
        if dt:
            self.dt = dt

        if inst_durations is None:
            return self

        if isinstance(inst_durations, InstructionDurations):
            self.duration_by_name.update(inst_durations.duration_by_name)
            self.duration_by_name_params.update(inst_durations.duration_by_name_params)
            self.duration_by_name_qubits.update(inst_durations.duration_by_name_qubits)
            self.duration_by_name_params_qubits.update(
                inst_durations.duration_by_name_params_qubits
            )
        else:
            for i, items in enumerate(inst_durations):
                if len(items) == 3:
                    inst_durations[i] = (items[0], None, items[1], items[2], "dt")
                elif len(items) == 4:
                    if isinstance(items[-1], str):
                        inst_durations[i] = (items[0], None, items[1], items[2], items[3])
                    else:
                        inst_durations[i] = (*items, "dt")  # set default unit
                elif len(items) != 5:
                    raise TranspilerError(
                        "Each entry of inst_durations dictionary must be "
                        "(inst_name, qubits, duration) or "
                        "(inst_name, qubits, duration, unit) or"
                        "(inst_name, params, qubits, duration) or"
                        "(inst_name, params, qubits, duration, unit)"
                    )

            for name, params, qubits, duration, unit in inst_durations:
                if isinstance(qubits, int):
                    qubits = [qubits]
                if isinstance(params, float):
                    params = [params]

                if params is None and qubits is None:
                    self.duration_by_name[name] = duration, unit
                elif params is None:
                    self.duration_by_name_qubits[(name, tuple(qubits))] = duration, unit
                elif qubits is None:
                    self.duration_by_name_params[(name, tuple(params))] = duration, unit
                else:
                    self.duration_by_name_params_qubits[(name, tuple(params), tuple(qubits))] = (
                        duration,
                        unit,
                    )

        return self

    def get(
        self,
        inst: Union[str, Instruction],
        params: Optional[Union[float, List[float]]] = None,
        qubits: Optional[Union[int, List[int], Qubit, List[Qubit]]] = None,
        unit: str = "dt",
    ) -> float:
        """Get the duration of the instruction with the name, parameters and qubits.

        Args:
            inst: An instruction or its name to be queried.
            params: Instruction parameters.
            qubits: Qubits or its indices that the instruction acts on.
            unit: The unit of duration to be returned. It must be 's' or 'dt'.

        Returns:
            float|int: The duration of the instruction on the qubits.

        Raises:
            TranspilerError: No duration is defined for the instruction.
        """
        if isinstance(inst, Barrier):
            return 0
        elif isinstance(inst, Delay):
            return self._convert_unit(inst.duration, inst.unit, unit)

        if isinstance(inst, Instruction):
            inst_name = inst.name
        else:
            inst_name = inst

        if isinstance(params, float):
            params = [params]

        if isinstance(qubits, (int, Qubit)):
            qubits = [qubits]

        if qubits and isinstance(qubits[0], Qubit):
            warnings.warn(
                "Querying an InstructionDurations object with a Qubit "
                "has been deprecated and will be removed in a future "
                "release. Instead, query using the integer qubit "
                "index.",
                DeprecationWarning,
                stacklevel=2,
            )
            qubits = [q.index for q in qubits]

        try:
            return self._get(inst_name, params, qubits, unit)
        except TranspilerError as ex:
            raise TranspilerError(
                f"Duration of {inst_name} with params {params} on" f"qubits {qubits} not found."
            ) from ex

    def _get(self, name: str, params: List[float], qubits: List[int], to_unit: str) -> float:
        """Get the duration of the instruction, priority with the most specific matching one."""
        if params is not None:
            params = tuple(params)
        if qubits is not None:
            qubits = tuple(qubits)

        if (name, params, qubits) in self.duration_by_name_params_qubits:
            duration, unit = self.duration_by_name_params_qubits[(name, params, qubits)]
        elif (name, params) in self.duration_by_name_params:
            duration, unit = self.duration_by_name_params[(name, params)]
        elif (name, qubits) in self.duration_by_name_qubits:
            duration, unit = self.duration_by_name_qubits[(name, qubits)]
        elif name in self.duration_by_name:
            duration, unit = self.duration_by_name[name]
        else:
            raise TranspilerError(
                f"No value is found for name={name}, params={params}, qubits={qubits}"
            )

        return self._convert_unit(duration, unit, to_unit)

    def _convert_unit(self, duration: float, from_unit: str, to_unit: str) -> float:
        if from_unit.endswith("s") and from_unit != "s":
            duration = apply_prefix(duration, from_unit)
            from_unit = "s"

        # assert both from_unit and to_unit in {'s', 'dt'}
        if from_unit == to_unit:
            return duration

        if self.dt is None:
            raise TranspilerError(
                f"dt is necessary to convert durations from '{from_unit}' to '{to_unit}'"
            )
        if from_unit == "s" and to_unit == "dt":
            if isinstance(duration, ParameterExpression):
                return duration / self.dt
            return duration_in_dt(duration, self.dt)
        elif from_unit == "dt" and to_unit == "s":
            return duration * self.dt
        else:
            raise TranspilerError(f"Conversion from '{from_unit}' to '{to_unit}' is not supported")

    def units_used(self) -> Set[str]:
        """Get the set of all units used in this instruction durations.

        Returns:
            Set of units used in this instruction durations.
        """
        units_used = set()
        for _, unit in self.duration_by_name_qubits.values():
            units_used.add(unit)
        for _, unit in self.duration_by_name.values():
            units_used.add(unit)
        return units_used


InstructionDurationsType = Union[
    List[Tuple[str, Optional[Iterable[float]], Optional[Iterable[int]], float, str]],
    List[Tuple[str, Optional[Iterable[float]], Optional[Iterable[int]], float]],
    InstructionDurations,
]
"""List of tuples representing (instruction name, params, qubits indices, duration)."""
