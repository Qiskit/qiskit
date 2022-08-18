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
Zero probability fidelity primitive
"""
from __future__ import annotations
import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.primitives.primitive_job import PrimitiveJob
from .base_fidelity import BaseFidelity
# from .fidelity_job import FidelityJob

class Fidelity(BaseFidelity):
    """
    Calculates the fidelity of two quantum circuits by measuring the zero probability outcome.
    """

    def __init__(
        self,
        sampler: Sampler,
        left_circuits: Sequence[QuantumCircuit] | None = None,
        right_circuits: Sequence[QuantumCircuit]| None = None,
    ) -> None:
        """
        Initializes the class to evaluate the fidelities defined as the state overlap
            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`,
        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``left_circuit`` and ``right_circuit``, respectively.
        Args:
            left_circuit: (Parametrized) quantum circuit :math:`|\psi\rangle`.
            right_circuit: (Parametrized) quantum circuit :math:`|\phi\rangle`.
            sampler: Sampler primitive instance.
        Raises:
            ValueError: left_circuit and right_circuit don't have the same number of qubits.
        """
        self.sampler = sampler
        super().__init__(left_circuits, right_circuits)

    def _call(
        self,
        circuits_list,
        values_list
    ) -> np.ndarray:
        """Run the state overlap (fidelity) calculation between 2
        parametrized circuits (left and right) for a specific set of parameter
        values (left and right).
        Args:
            left_circuits: (Parametrized) quantum circuit preparing :math:`|\psi\rangle`.
                          If a list of circuits is sent, only the first circuit will be
                          taken into account.
            right_circuits: (Parametrized) quantum circuit preparing :math:`|\phi\rangle`.
                          If a list of circuits is sent, only the first circuit will be
                          taken into account.
            left_values: Numerical parameters to be bound to the left circuit.
            right_values: Numerical parameters to be bound to the right circuit.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The result for the fidelity calculation.
        """

        if circuits_list is None:
            raise ValueError(
                "The left and right circuits must be defined to "
                "calculate the state overlap. "
            )

        if len(values_list) > 0:
            if len(values_list) == 2 and values_list[0].shape[0] != values_list[1].shape[0]:
                raise ValueError(
                    f"The number of left parameters (currently {values_list[0].shape[0]})"
                    "has to be equal to the number of right parameters."
                    f"(currently {values_list[1].shape[0]})"
                )
            values = np.hstack(values_list)
            job = self.sampler.run(circuits=circuits_list, parameter_values=values)
        else:
            job = self.sampler.run(circuits=circuits_list)

        result = job.result()

        # if error mitigation is added in the future, we will have to handle
        # negative values in some way (e.g. clipping to zero)
        overlaps = [prob_dist.get(0, 0) for prob_dist in result.quasi_dists]
        return np.array(overlaps)

    def evaluate(self,
        left_circuits: Sequence[QuantumCircuit],
        right_circuits: Sequence[QuantumCircuit],
        left_values: Sequence[Sequence[float]] | None = None,
        right_values: Sequence[Sequence[float]] | None = None
    ) -> np.ndarray:

        circuit_indices = self._set_circuits(left_circuits, right_circuits)
        circuit_mapping = map(self._circuits.__getitem__, circuit_indices)
        circuits_list = list(circuit_mapping)

        values_list = []
        for values, side, circuits in zip([left_values, right_values], ["left", "right"],
                                [left_circuits, right_circuits]):
            values = self._check_values(values, side, circuits)
            if values is not None:
                values_list.append(values)

        overlaps = self._call(circuits_list, values_list)
        return overlaps




