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

from abc import abstractmethod
import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.primitives.fidelity import BaseFidelity


from typing import Callable, List, Union

SamplerFactory = Callable[[List[QuantumCircuit]], Sampler]


class Fidelity(BaseFidelity):
    """
    Calculates the fidelity of two quantum circuits by measuring the zero probability outcome.
    """

    def __init__(
        self,
        left_circuit: QuantumCircuit,
        right_circuit: QuantumCircuit,
        sampler_factory: SamplerFactory,
    ):
        """
        Initializes the class to evaluate the fidelities defined as the state overlap
            |<left_circuit(x), right_circuit(y)>|^2,
        where x and y are parametrizations of the circuits.
        Args:
            - left_circuit: (Parametrized) quantum circuit
            - right_circuit: (Parametrized) quantum circuit
            - sampler_factory: Optional partial sampler used as a backend
        Raises:
            - ValueError: left_circuit and right_circuit don't have the same number of qubits
        """
        super().__init__(left_circuit, right_circuit)

        circuit = self._left_circuit.compose(self._right_circuit.inverse())
        circuit.measure_all()

        self.sampler = sampler_factory([circuit])

    def compute(
        self,
        values_left: Union[np.ndarray, List[np.ndarray]],
        values_right: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[float, List[float]]:

        values_left = np.atleast_2d(values_left)
        values_right = np.atleast_2d(values_right)

        if values_left.shape[0] != values_right.shape[0]:
            raise ValueError(
                f"The number of left parameters (currently {values_left.shape[0]}) has to be equal to the number of right parameters (currently {values_right.shape[0]})"
            )

        values = np.hstack([values_left, values_right])
        result = self.sampler(circuit_indices=[0] * len(values), parameter_values=values)

        overlaps = [prob_dist.get(0, 0) for prob_dist in result.quasi_dists]
        return np.array(overlaps)
