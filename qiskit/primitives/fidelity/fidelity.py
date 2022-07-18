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
from .base_fidelity import BaseFidelity


class Fidelity(BaseFidelity):
    """
    Calculates the fidelity of two quantum circuits by measuring the zero probability outcome.
    """

    def __init__(
        self,
        sampler: Sampler,
        left_circuit: QuantumCircuit | None = None,
        right_circuit: QuantumCircuit | None = None,
    ) -> None:
        r"""
        Initializes the class to evaluate the fidelities defined as the state overlap
            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`,
        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``left_circuit`` and ``right_circuit``, respectively.
        Args:
            left_circuit: (Parametrized) quantum circuit :math:`|\psi\rangle`.
            right_circuit: (Parametrized) quantum circuit :math:`|\phi\rangle`.
            sampler_factory: Partial sampler used as a backend.
        Raises:
            ValueError: left_circuit and right_circuit don't have the same number of qubits.
        """
        self.sampler = sampler
        super().__init__(left_circuit, right_circuit)

    def __call__(
        self,
        values_left: np.ndarray | list[np.ndarray] | None = None,
        values_right: np.ndarray | list[np.ndarray] | None = None,
    ) -> np.ndarray:

        values_list = []

        if self._left_circuit is None or self._right_circuit is None:
            raise ValueError(
                f"The left and right circuits must be defined to"
                f"calculate the state overlap. "
                f"Please use .set_circuits(left_circuit, right_circuit)"
            )

        for values, side in zip([values_left, values_right], ["left", "right"]):
            values = self._check_values(values, side)
            if values is not None:
                values_list.append(values)

        if len(values_list) > 0:
            if len(values_list) == 2 and values_list[0].shape[0] != values_list[1].shape[0]:
                raise ValueError(
                    f"The number of left parameters (currently {values_list[0].shape[0]})"
                    "has to be equal to the number of right parameters."
                    f"(currently {values_list[1].shape[0]})"
                )
            values = np.hstack(values_list)
            result = self.sampler(circuits=[0] * len(values), parameter_values=values)
        else:
            result = self.sampler(circuits=[0])

        # if error mititgation is added in the future, we will have to handle
        # negative values in some way (e.g. clipping to zero)
        overlaps = [prob_dist.get(0, 0) for prob_dist in result.quasi_dists]

        return np.array(overlaps)

    def set_circuits(self, left_circuit: QuantumCircuit, right_circuit: QuantumCircuit):
        """
        Fix the circuits for the fidelity to be computed of.
        Args:
        - left_circuit: (Parametrized) quantum circuit
        - right_circuit: (Parametrized) quantum circuit
        """
        super().set_circuits(left_circuit=left_circuit, right_circuit=right_circuit)

        circuit = self._left_circuit.compose(self._right_circuit.inverse())
        circuit.measure_all()

        # in the future this should be self.sampler.add_circuit(circuit)
        # Careful! because add_circuits doesn't exist yet, calling this method
        # twice will make it store the result of a sampler call in self.sampler.
        self.sampler = self.sampler([circuit])
