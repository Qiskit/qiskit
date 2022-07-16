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
Base fidelity primitive
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class BaseFidelity(ABC):
    """
    Defines the interface to calculate fidelities.
    """

    def __init__(
        self,
        left_circuit: QuantumCircuit,
        right_circuit: QuantumCircuit,
    ) -> None:
        r"""Initializes the class to evaluate the fidelities defined as

            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`,

        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``left_circuit`` and ``right_circuit``, respectively.

        Args:
            left_circuit: (Parametrized) quantum circuit preparing :math:`|\psi\rangle`.
            right_circuit: (Parametrized) quantum circuit preparing :math:`|\phi\rangle`.

        Raises:
            ValueError: ``left_circuit`` and ``right_circuit`` don't have the same number of qubits.
        """

        if left_circuit.num_qubits != right_circuit.num_qubits:
            raise ValueError(
                f"The number of qubits for the left circuit ({left_circuit.num_qubits})"
                f"and right circuit ({right_circuit.num_qubits}) do not coincide."
            )

        self._left_circuit = left_circuit
        self._right_circuit = right_circuit

        # Reassigning parameters to make sure that left and right parameters are independent
        # even in case that left_circuit == right_circuit
        left_parameters = ParameterVector("x", left_circuit.num_parameters)
        self._left_circuit = left_circuit.assign_parameters(left_parameters)

        right_parameters = ParameterVector("y", right_circuit.num_parameters)
        self._right_circuit = right_circuit.assign_parameters(right_parameters)

    @abstractmethod
    def compute(
        self,
        values_left: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        values_right: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> np.ndarray:
        """Compute the overlap of two quantum states bound by the
        parametrizations values_left and values_right.

        Args:
            values_left: Numerical parameters to be bound to the left circuit
            values_right: Numerical parameters to be bound to the right circuit

        Returns:
            The overlap of two quantum states defined by two parametrized circuits.
        """
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass
