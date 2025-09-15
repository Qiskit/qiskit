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
from __future__ import annotations
from typing import Optional, List, Tuple, Union, Iterable

import qiskit.circuit
from qiskit.circuit import Barrier, Delay, Instruction, ParameterExpression
from qiskit.circuit.duration import duration_in_dt
from qiskit.providers import Backend
from qiskit.providers.backend import BackendV2
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.units import apply_prefix


class InstructionDurations:
    """
    Helper class to provide durations of instructions for scheduling.

    It stores durations (gate lengths) and dt to be used at the scheduling stage of transpiling.
    It can be constructed from ``backend`` or ``instruction_durations``,
    which is an argument of :func:`transpile`. The duration of an instruction depends on the
    instruction (given by name), the qubits, and optionally the parameters of the instruction.
    Note that these fields are used as keys in dictionaries that are used to retrieve the
    instruction durations. Therefore, users must use the exact same parameter value to retrieve
    an instruction duration as the value with which it was added.

    Attributes:
        duration_by_name: Dictionary mapping instruction names to (duration, unit) tuples.
        duration_by_name_qubits: Dictionary mapping (name, qubits) to (duration, unit) tuples.
        duration_by_name_qubits_params: Dictionary mapping (name, qubits, params) to
            (duration, unit) tuples.
        dt: Sampling duration in seconds for unit conversion.

    .. note::

        - When ``qubits`` is ``None``, the duration applies to all qubits as a default.
        - When ``parameters`` is ``None``, the duration applies to any parameter values.
        - More specific entries (with qubits/parameters) take priority over general ones.

    Example:

        >>> # Create InstructionDurations with various tuple formats
        >>> durations = InstructionDurations([
        ...     ('x', None, 160, 'dt'),           # x gate on any qubit: 160 dt
        ...     ('sx', [0], 80, 'dt'),            # sx gate on qubit 0 only: 80 dt
        ...     ('cx', [0, 1], 800, 'dt'),        # cx gate on qubits 0,1: 800 dt
        ...     ('measure', None, 5000, 'dt'),    # measure on any qubit: 5000 dt
        ...     ('rx', [0], 150, [1.5708], 'dt'), # rx(π/2) on qubit 0: 150 dt
        ...     ('ry', [1], 120, None, 'dt')      # ry with any parameters on qubit 1: 120 dt
        ... ], dt=1e-7)
        >>>
        >>> durations.duration_by_name
        {'x': (160, 'dt'), 'measure': (5000, 'dt')}
        >>> durations.duration_by_name_qubits
        {('sx', (0,)): (80, 'dt'), ('cx', (0, 1)): (800, 'dt'), ('ry', (1,)): (120, 'dt')}
        >>> durations.duration_by_name_qubits_params
        {('rx', (0,), (1.5708,)): (150, 'dt')}
        >>>
        >>> durations.get("x", 0)      # Uses default from duration_by_name (qubits=None)
        160.0
        >>> durations.get("x", 5)      # Also uses default (qubits=None applies to all qubits)
        160.0
        >>> durations.get("sx", 0)     # Uses specific qubit duration
        80.0
        >>> durations.get("ry", 1)     # Uses specific qubit duration with parameters=None
        120.0
        >>> durations.get("rx", [0], parameters=[1.5708])  # Uses parameterized duration
        150.0

    """

    def __init__(
        self, instruction_durations: "InstructionDurationsType" | None = None, dt: float = None
    ):
        """
        Initialize an InstructionDurations object.

        Args:
            instruction_durations:
                A list of tuples in one of the following formats:
                    - (inst_name, qubits, duration)
                    - (inst_name, qubits, duration, unit)
                    - (inst_name, qubits, duration, parameters)
                    - (inst_name, qubits, duration, parameters, unit)
                    - An existing InstructionDurations object.

                Where:
                    - inst_name (str): Instruction name (e.g., 'x', 'cx', 'measure').
                    - qubits (int | list[int] | None): Target qubits.
                    - duration (float): Duration value.
                    - parameters (list[float] | None): Parameters for parameterized instructions.
                    - unit (str): Time unit ('dt', 's', 'ms', 'us', 'ns'), defaults to 'dt'.

            dt: Sampling duration in seconds.
        """
        self.duration_by_name: dict[str, tuple[float, str]] = {}
        self.duration_by_name_qubits: dict[tuple[str, tuple[int, ...]], tuple[float, str]] = {}
        self.duration_by_name_qubits_params: dict[
            tuple[str, tuple[int, ...], tuple[float, ...]], tuple[float, str]
        ] = {}
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
        for k, v in self.duration_by_name_qubits.items():
            string += k[0] + str(k[1])
            string += ": "
            string += str(v[0]) + " " + v[1]
            string += "\n"
        return string

    @classmethod
    def from_backend(cls, backend: Backend):
        """Construct an :class:`InstructionDurations` object from the backend.

        Args:
            backend: backend from which durations (gate lengths) and dt are extracted.

        Returns:
            InstructionDurations: The InstructionDurations constructed from backend.

        Raises:
            TranspilerError: If dt and dtm is different in the backend.
            TypeError: If the backend is the wrong type
        """
        # All durations in seconds in gate_length
        if isinstance(backend, BackendV2):
            return backend.target.durations()
        raise TypeError("Unsupported backend type: {backend}")

    def update(self, inst_durations: InstructionDurationsType | None, dt: float = None):
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
            self.duration_by_name_qubits.update(inst_durations.duration_by_name_qubits)
            self.duration_by_name_qubits_params.update(
                inst_durations.duration_by_name_qubits_params
            )
        else:
            for i, items in enumerate(inst_durations):
                if not isinstance(items[-1], str):
                    items = (*items, "dt")  # set default unit

                if len(items) == 4:  # (inst_name, qubits, duration, unit)
                    inst_durations[i] = (*items[:3], None, items[3])
                else:
                    inst_durations[i] = items

                # assert (inst_name, qubits, duration, parameters, unit)
                if len(inst_durations[i]) != 5:
                    raise TranspilerError(
                        "Each entry of inst_durations dictionary must be "
                        "(inst_name, qubits, duration) or "
                        "(inst_name, qubits, duration, unit) or"
                        "(inst_name, qubits, duration, parameters) or"
                        "(inst_name, qubits, duration, parameters, unit) "
                        f"received {inst_durations[i]}."
                    )

                if inst_durations[i][2] is None:
                    raise TranspilerError(f"None duration for {inst_durations[i]}.")

            for name, qubits, duration, parameters, unit in inst_durations:
                if isinstance(qubits, int):
                    qubits = [qubits]

                if isinstance(parameters, (int, float)):
                    parameters = [parameters]

                if qubits is None:
                    self.duration_by_name[name] = duration, unit
                elif parameters is None:
                    self.duration_by_name_qubits[(name, tuple(qubits))] = duration, unit
                else:
                    key = (name, tuple(qubits), tuple(parameters))
                    self.duration_by_name_qubits_params[key] = duration, unit

        return self

    def get(
        self,
        inst: str | qiskit.circuit.Instruction,
        qubits: int | list[int],
        unit: str = "dt",
        parameters: list[float] | None = None,
    ) -> float:
        """Get the duration of the instruction with the name, qubits, and parameters.

        Some instructions may have a parameter dependent duration.

        Args:
            inst: An instruction or its name to be queried.
            qubits: Qubit indices that the instruction acts on.
            unit: The unit of duration to be returned. It must be 's' or 'dt'.
            parameters: The value of the parameters of the desired instruction.

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

        if isinstance(qubits, int):
            qubits = [qubits]

        try:
            return self._get(inst_name, qubits, unit, parameters)
        except TranspilerError as ex:
            raise TranspilerError(
                f"Duration of {inst_name} on qubits {qubits} is not found."
            ) from ex

    def _get(
        self,
        name: str,
        qubits: list[int],
        to_unit: str,
        parameters: Iterable[float] | None = None,
    ) -> float:
        """Get the duration of the instruction with the name, qubits, and parameters."""
        if name == "barrier":
            return 0

        if parameters is not None:
            key = (name, tuple(qubits), tuple(parameters))
        else:
            key = (name, tuple(qubits))

        if key in self.duration_by_name_qubits_params:
            duration, unit = self.duration_by_name_qubits_params[key]
        elif key in self.duration_by_name_qubits:
            duration, unit = self.duration_by_name_qubits[key]
        elif name in self.duration_by_name:
            duration, unit = self.duration_by_name[name]
        else:
            raise TranspilerError(f"No value is found for key={key}")

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

    def units_used(self) -> set[str]:
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
    List[Tuple[str, Optional[Iterable[int]], float, Optional[Iterable[float]], str]],
    List[Tuple[str, Optional[Iterable[int]], float, Optional[Iterable[float]]]],
    List[Tuple[str, Optional[Iterable[int]], float, str]],
    List[Tuple[str, Optional[Iterable[int]], float]],
    InstructionDurations,
]
"""
Type alias for instruction durations.
"""
