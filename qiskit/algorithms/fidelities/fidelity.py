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
from typing import Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from .base_fidelity import BaseFidelity


class Fidelity(BaseFidelity):
    """
    Calculates the fidelity of two quantum circuits by measuring the zero probability outcome.
    """

    def __init__(
        self,
        sampler: Sampler,
    ) -> None:
        """
        Initializes the class to evaluate the fidelities defined as the state overlap
            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`,
        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``left_circuit`` and ``right_circuit``, respectively.
        Args:
            sampler: Sampler primitive instance.
        """
        self._sampler = sampler
        super().__init__()

    def _preprocess_inputs(
        self,
        left_circuits: Sequence[QuantumCircuit],
        right_circuits: Sequence[QuantumCircuit],
        left_values: Sequence[Sequence[float]] | None = None,
        right_values: Sequence[Sequence[float]] | None = None,
    ) -> tuple(list[QuantumCircuit], list[list[float]]):
        """Preprocess circuits and parameter values.

         Args:
            left_circuits: (Parametrized) quantum circuit preparing :math:`|\psi\rangle`.
                          If a list of circuits is sent, only the first circuit will be
                          taken into account.
            right_circuits: (Parametrized) quantum circuit preparing :math:`|\phi\rangle`.
                          If a list of circuits is sent, only the first circuit will be
                          taken into account.
            left_values: Numerical parameters to be bound to the left circuit.
            right_values: Numerical parameters to be bound to the right circuit.

        Returns:
            Preprocessed circuits and parameter values.

        Raises:
            ValueError: The number of left parameters has to be equal to the number of
                        right parameters.
        """

        # _set_circuits returns indices in list of cached circuits
        circuit_indices = self._set_circuits(left_circuits, right_circuits)
        circuit_mapping = map(self._circuits.__getitem__, circuit_indices)
        # final list of circuits that will be evaluated
        circuits_list = list(circuit_mapping)

        left_values = self._check_values(left_values, "left", left_circuits)
        right_values = self._check_values(right_values, "right", right_circuits)

        values_list = []
        if right_values is not None or left_values is not None:
            if right_values is None:
                values_list = left_values
            elif left_values is None:
                values_list = right_values
            else:
                for (left_val, right_val) in zip(left_values, right_values):
                    if len(left_val) != len(right_val):
                        raise ValueError(
                            f"The number of left parameters (currently {len(left_val)})"
                            f"has to be equal to the number of right parameters."
                            f"(currently {len(right_val)})"
                        )
                    values_list.append(left_val + right_val)

        return circuits_list, values_list

    def evaluate(
        self,
        left_circuits: Sequence[QuantumCircuit],
        right_circuits: Sequence[QuantumCircuit],
        left_values: Sequence[Sequence[float]] | None = None,
        right_values: Sequence[Sequence[float]] | None = None,
        **run_options,
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
            The result of the fidelity calculation.

        Raises:
            ValueError: At least one left and right circuit must be defined.
        """

        circuits_list, values_list = self._preprocess_inputs(
            left_circuits, right_circuits, left_values, right_values
        )

        if circuits_list is None:
            raise ValueError(
                "At least one left and right circuit must be defined to "
                "calculate the state overlap. "
            )

        if len(values_list) > 0:
            job = self._sampler.run(
                circuits=circuits_list, parameter_values=values_list, **run_options
            )
        else:
            job = self._sampler.run(circuits=circuits_list, **run_options)

        result = job.result()

        # if error mitigation is added in the future, we will have to handle
        # negative values in some way (e.g. clipping to zero)
        overlaps = [prob_dist.get(0, 0) for prob_dist in result.quasi_dists]
        return np.array(overlaps)
