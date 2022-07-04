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


from typing import Callable, List, Union, Optional

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
    ) -> None:
        """
        Initializes the class to evaluate the fidelities defined as the state overlap
            :math:`|\braket{\psi(x)} {\phi(y)}|^2`,
        where x and y are parametrizations of the circuits :math:`\psi` and :math:`\phi`.
        Args:
            - left_circuit: (Parametrized) quantum circuit :math:`\psi`
            - right_circuit: (Parametrized) quantum circuit :math:`\phi`
            - sampler_factory: Partial sampler used as a backend
        Raises:
            - ValueError: left_circuit and right_circuit don't have the same number of qubits
        """
        super().__init__(left_circuit, right_circuit)

        circuit = self._left_circuit.compose(self._right_circuit.inverse())
        circuit.measure_all()

        self.sampler = sampler_factory([circuit])

    def compute(
        self,
        values_left: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        values_right: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> np.ndarray:

        values_list = []

        if values_left is None:
            if self._left_circuit.num_parameters != 0:
                raise ValueError(
                    f"`values_left` cannot be `None` because the left circuit has {self._left_circuit.num_parameters} free parameters."
                )
        else:
            values_list.append(np.atleast_2d(values_left))

        if values_right is None:
            if self._right_circuit.num_parameters != 0:
                raise ValueError(
                    f"`values_left` cannot be `None` because the left circuit has {self._left_circuit.num_parameters} free parameters."
                )
        else:
            values_list.append(np.atleast_2d(values_right))

        if len(values_list) > 0:
            if len(values_list) == 2 and values_list[0].shape[0] != values_list[1].shape[0]:
                raise ValueError(
                    f"The number of left parameters (currently {values_list[0].shape[0]}) has to be equal to the number of right parameters (currently {values_list[1].shape[0]})"
                )
            values = np.hstack(values_list)
            result = self.sampler(circuits=[0] * len(values), parameter_values=values)
        else:
            result = self.sampler(circuits=[0])

        overlaps = [prob_dist.get(0, 0) for prob_dist in result.quasi_dists]

        return np.array(overlaps)
