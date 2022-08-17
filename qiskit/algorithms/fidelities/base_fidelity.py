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

from __future__ import annotations
from abc import ABC, abstractmethod
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

        self._circuit = None
        if left_circuit is None or right_circuit is None:
            self._left_circuit = None
            self._right_circuit = None
            self._left_parameters = None
            self._right_parameters = None
        else:
            self._set_circuits(left_circuit, right_circuit)

    def run(
        self,
        left_circuit: Sequence[QuantumCircuit] | None = None,
        right_circuit: Sequence[QuantumCircuit] | None = None,
        left_values: Sequence[Sequence[float]] | None = None,
        right_values: Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> FidelityJob:
        """Compute the overlap of two quantum states bound by the
        parametrizations left_values and right_values.

        Args:
            left_values: Numerical parameters to be bound to the left circuit.
            right_values: Numerical parameters to be bound to the right circuit.
            left_circuit: (Parametrized) quantum circuit preparing :math:`|\psi\rangle`.
            right_circuit: (Parametrized) quantum circuit preparing :math:`|\phi\rangle`.

        Returns:
            The overlap of two quantum states defined by two parametrized circuits.
        """
        return self._run(left_circuit,
                         right_circuit,
                         left_values,
                         right_values,
                         **run_options)

    @abstractmethod
    def _run(
        self,
        left_circuit: Sequence[QuantumCircuit] | None = None,
        right_circuit: Sequence[QuantumCircuit] | None = None,
        left_values: Sequence[Sequence[float]] | None = None,
        right_values: Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> FidelityJob:
        """Compute the overlap of two quantum states bound by the
            parametrizations left_values and right_values.
            Args:
                left_circuit: (Parametrized) quantum circuit preparing :math:`|\psi\rangle`.
                          If a list of circuits is sent, only the first circuit will be
                          taken into account.
                right_circuit: (Parametrized) quantum circuit preparing :math:`|\phi\rangle`.
                              If a list of circuits is sent, only the first circuit will be
                              taken into account.
                left_values: Numerical parameters to be bound to the left circuit.
                right_values: Numerical parameters to be bound to the right circuit.
                run_options: Backend runtime options used for circuit execution.

            Returns:
                The job object for the fidelity calculation.
            """
        raise NotImplementedError()

    def _check_values(
        self, values: np.ndarray | list[np.ndarray] | None, side: str
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
                    f"`values_{side}` cannot be `None` because the {side} circuit has "
                    f"{circuit.num_parameters} free parameters. {circuit}"
                )
            return None
        else:
            return np.atleast_2d(values)

    def _set_circuits(
        self,
        left_circuit: QuantumCircuit | None = None,
        right_circuit: QuantumCircuit | None = None,
    ):
        """
        Fix the circuits for the fidelity to be computed of.
        Args:
            left_circuit: (Parametrized) quantum circuit
            right_circuit: (Parametrized) quantum circuit

        Raises:
            ValueError: ``left_circuit`` and ``right_circuit`` don't have the same number of qubits.
        """
        if left_circuit is not None and right_circuit is not None:
            self._check_qubits_mismatch(left_circuit, right_circuit)
            self._set_left_circuit(left_circuit)
            self._set_right_circuit(right_circuit)

        elif left_circuit is not None:
            self._check_qubits_mismatch(left_circuit, self._right_circuit)
            self._set_left_circuit(left_circuit)
        elif right_circuit is not None:
            self._check_qubits_mismatch(self._left_circuit, right_circuit)
            self._set_right_circuit(right_circuit)
        else:
            raise ValueError(
                "At least one of the arguments `left_circuit` or `right_circuit` must not be `None`."
            )

    def _set_left_circuit(self, circuit: QuantumCircuit) -> None:
        """
        Fix the left circuit. If `check_num_qubits` the number of qubits are compared
        to the right circuit.
        """
        self._left_parameters = ParameterVector("x", circuit.num_parameters)
        self._left_circuit = circuit.assign_parameters(self._left_parameters)

    def _set_right_circuit(self, circuit: QuantumCircuit) -> None:
        """
        Fix the right circuit. If `check_num_qubits` the number of qubits are compared
        to the left circuit.
        """
        self._right_parameters = ParameterVector("y", circuit.num_parameters)
        self._right_circuit = circuit.assign_parameters(self._right_parameters)

    def _check_qubits_mismatch(
        self, left_circuit: QuantumCircuit, right_circuit: QuantumCircuit
    ) -> None:
        if left_circuit is not None and right_circuit is not None:
            if left_circuit.num_qubits != right_circuit.num_qubits:
                raise ValueError(
                    f"The number of qubits for the left circuit ({left_circuit.num_qubits}) \
                        and right circuit ({right_circuit.num_qubits}) do not coincide."
                )
