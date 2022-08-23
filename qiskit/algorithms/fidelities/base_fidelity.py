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
from abc import ABC
from typing import Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class BaseFidelity(ABC):
    """
    Defines the interface to calculate fidelities.
    """

    def __init__(
        self,
    ) -> None:
        r"""Initializes the class to evaluate the fidelities defined as

            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`,

        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``left_circuit`` and ``right_circuit``, respectively.
        """

        self._circuits = []
        self._circuit_ids = {}

        self._left_parameters = []
        self._right_parameters = []

    def _check_values(
        self, values: Sequence[Sequence[float]] | None, side: str, circuits: QuantumCircuit
    ) -> list[list[float]] | None:
        """
        Check whether the passed values match the shape of the parameters of the circuit on the side
        provided.

        Returns a 2D-Array if values match, `None` if no parameters are passed and raises an error if
        the shapes don't match.
        """

        # Support ndarray
        if isinstance(values, np.ndarray):
            values = values.tolist()

        if values is None:
            for circuit in circuits:
                if circuit.num_parameters != 0:
                    raise ValueError(
                        f"`values_{side}` cannot be `None` because the {side} circuit has "
                        f"{circuit.num_parameters} free parameters."
                    )
            return None
        else:
            # ensure 2d list
            if not isinstance(values, list):
                values = [values]
            if len(values) > 0 and not isinstance(values[0], list):
                values = [values]
            return values

    def _check_qubits_mismatch(
        self, left_circuit: QuantumCircuit, right_circuit: QuantumCircuit
    ) -> None:
        """
        Check that the number of qubits of the left and right circuit matches
        Args:
            left_circuit: (Parametrized) quantum circuit
            right_circuit: (Parametrized) quantum circuit

        Raises:
            ValueError: ``left_circuit`` and ``right_circuit`` don't have the same number of qubits.
        """

        if left_circuit is not None and right_circuit is not None:
            if left_circuit.num_qubits != right_circuit.num_qubits:
                raise ValueError(
                    f"The number of qubits for the left circuit ({left_circuit.num_qubits}) \
                        and right circuit ({right_circuit.num_qubits}) do not coincide."
                )

    def _set_circuits(
        self,
        left_circuits: Sequence[QuantumCircuit] | None = None,
        right_circuits: Sequence[QuantumCircuit] | None = None,
    ) -> np.ndarray:
        """
        Fix the circuits for the fidelity to be computed of.
        Args:
            left_circuit: (Parametrized) quantum circuit
            right_circuit: (Parametrized) quantum circuit
        """

        if not len(left_circuits) == len(right_circuits):
            raise ValueError(
                f"The number of left ({len(left_circuits)}) \
                        and right circuits ({len(right_circuits)}) do not coincide."
            )

        circuit_indices = []
        for (left_circuit, right_circuit) in zip(left_circuits, right_circuits):

            index = self._circuit_ids.get((id(left_circuit), id(right_circuit)))

            if index is not None:
                # the composed circuit already exists
                circuit_indices.append(index)
            else:
                # create new circuit
                self._check_qubits_mismatch(left_circuit, right_circuit)

                left_parameters = ParameterVector("x", left_circuit.num_parameters)
                self._left_parameters.append(left_parameters)
                parametrized_left_circuit = left_circuit.assign_parameters(left_parameters)

                right_parameters = ParameterVector("y", right_circuit.num_parameters)
                self._right_parameters.append(right_parameters)
                parametrized_right_circuit = right_circuit.assign_parameters(right_parameters)

                circuit = parametrized_left_circuit.compose(parametrized_right_circuit.inverse())
                circuit.measure_all()

                self._circuit_ids[id(left_circuit), id(right_circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                circuit_indices.append(len(self._circuits) - 1)

        return circuit_indices
