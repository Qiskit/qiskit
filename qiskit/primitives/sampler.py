# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Sampler class
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution

from .base_sampler import BaseSampler
from .sampler_result import SamplerResult
from .utils import final_measurement_mapping, init_circuit


class Sampler(BaseSampler):
    """
    Sampler class
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        parameters: Iterable[Iterable[Parameter]] | None = None,
    ):
        """
        Args:
            circuits: circuits to be executed
            parameters: Parameters of each of the quantum circuits.
                Defaults to ``[circ.parameters for circ in circuits]``.

        Raises:
            QiskitError: if some classical bits are not used for measurements.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        circuits = [init_circuit(circuit) for circuit in circuits]
        q_c_mappings = [final_measurement_mapping(circuit) for circuit in circuits]
        self._qargs_list = []
        for circuit, q_c_mapping in zip(circuits, q_c_mappings):
            if set(range(circuit.num_clbits)) != set(q_c_mapping.values()):
                raise QiskitError(
                    "some classical bits are not used for measurements."
                    f" the number of classical bits {circuit.num_clbits},"
                    f" the used classical bits {set(q_c_mapping.values())}."
                )
            c_q_mapping = sorted((c, q) for q, c in q_c_mapping.items())
            self._qargs_list.append([q for _, q in c_q_mapping])
        circuits = [circuit.remove_final_measurements(inplace=False) for circuit in circuits]
        super().__init__(circuits, parameters)
        self._is_closed = False

    def __call__(
        self,
        circuit_indices: Sequence[int] | None = None,
        parameter_values: Sequence[Sequence[float]] | Sequence[float] | None = None,
        **run_options,
    ) -> SamplerResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()
        if parameter_values and not isinstance(parameter_values[0], (np.ndarray, Sequence)):
            parameter_values = cast("Sequence[float]", parameter_values)
            parameter_values = [parameter_values]
        if circuit_indices is None:
            circuit_indices = list(range(len(self._circuits)))
        if parameter_values is None:
            parameter_values = [[]] * len(circuit_indices)
        if len(circuit_indices) != len(parameter_values):
            raise QiskitError(
                f"The number of circuit indices ({len(circuit_indices)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        bound_circuits_qargs = []
        for i, value in zip(circuit_indices, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )
            bound_circuits_qargs.append(
                (
                    self._circuits[i].bind_parameters(dict(zip(self._parameters[i], value))),
                    self._qargs_list[i],
                )
            )
        probabilities = [
            Statevector(circ).probabilities(qargs=qargs) for circ, qargs in bound_circuits_qargs
        ]
        quasis = [QuasiDistribution(dict(enumerate(p))) for p in probabilities]

        return SamplerResult(quasis, [{}] * len(circuit_indices))

    def close(self):
        self._is_closed = True
