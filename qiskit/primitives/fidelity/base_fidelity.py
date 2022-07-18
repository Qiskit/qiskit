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
        left_circuit: QuantumCircuit | None = None,
        right_circuit: QuantumCircuit | None = None,
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

        if left_circuit is None or right_circuit is None:
            self._left_circuit = None
            self._right_circuit = None
            self._left_parameters = None
            self._right_parameters = None
        else:
            self.set_circuits(left_circuit, right_circuit)

    @abstractmethod
    def __call__(
        self,
        values_left: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        values_right: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> np.ndarray:
        """Compute the overlap of two quantum states bound by the
        parametrizations values_left and values_right.

        Args:
            values_left: Numerical parameters to be bound to the left circuit.
            values_right: Numerical parameters to be bound to the right circuit.

        Returns:
            The overlap of two quantum states defined by two parametrized circuits.
        """
        raise NotImplementedError

    def _check_values(
        self, values: np.ndarray | List[np.ndarray] | None, side: str
    ) -> np.ndarray | None:
        """
        Check whether the passed values match the shape of the parameters of the circuit on the side
        provided.

        Returns a 2D-Array if values match, `None` if no parameters are passed and raises an error if
        the shapes don't match.
        """
        if side == "left":
            circuit = self._left_circuit
        else:
            circuit = self._right_circuit

        if values is None:
            if circuit.num_parameters != 0:
                raise ValueError(
                    f"`values_{side}` cannot be `None` because the {side} circuit has"
                    f"{circuit.num_parameters} free parameters."
                )
            return None
        else:
            return np.atleast_2d(values)

    @abstractmethod
    def set_circuits(self,
                     left_circuit: QuantumCircuit,
                     right_circuit: QuantumCircuit):
         """
         Fix the circuits for the fidelity to be computed of.
         Args:
             - left_circuit: (Parametrized) quantum circuit
             - right_circuit: (Parametrized) quantum circuit
         """
         if left_circuit.num_qubits != right_circuit.num_qubits:
             raise ValueError(
                 f"The number of qubits for the left circuit ({left_circuit.num_qubits}) \
                 and right circuit ({right_circuit.num_qubits}) do not coincide."
             )
         # Assigning parameter arrays to the two circuits
         self._left_parameters = ParameterVector("x", left_circuit.num_parameters)
         self._left_circuit = left_circuit.assign_parameters(self._left_parameters)

         self._right_parameters = ParameterVector("y", right_circuit.num_parameters)
         self._right_circuit = right_circuit.assign_parameters(self._right_parameters)
