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
Base fidelity interface
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Mapping
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class BaseFidelity(ABC):
    """
    An interface to calculate fidelities (state overlaps) for pairs of
    (parametrized) quantum circuits.
    """

    def __init__(
        self,
    ) -> None:
        """Initializes the class to evaluate fidelities."""

        self._circuits: Sequence[QuantumCircuit] = []
        self._parameter_values: Sequence[Sequence[float]] = []

        # use cache for preventing unnecessary circuit compositions
        self._circuit_cache: Mapping[(int, int), QuantumCircuit] = {}

        self._left_parameters = []
        self._right_parameters = []

    def _check_values_shape(
        self,
        values: Sequence[Sequence[float]] | None,
        circuits: QuantumCircuit,
        label: str | None = " ",
    ) -> list[list[float]] | None:
        """
        Checks whether the passed values match the shape of the parameters
        of the corresponding circuits and formats values to 2D list.

        Args:
            values: parameter values corresponding to the circuits to be checked
            circuits: list of circuits to be checked
            label: optional label to allow for circuit identification in error message

        Returns:
            Returns a 2D list if values match, `None` if no parameters are passed

        Raises:
            ValueError: if the number of parameter values doesn't match the number of
                        circuit parameters
        """

        # Support ndarray
        if isinstance(values, np.ndarray):
            values = values.tolist()

        if values is None:
            for circuit in circuits:
                if circuit.num_parameters != 0:
                    raise ValueError(
                        f"`values_{label}` cannot be `None` because circuit_{label} has "
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

    def _check_qubits_mismatch(self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit) -> None:
        """
        Checks that the number of qubits of 2 circuits matches.
        Args:
            circuit_1: (Parametrized) quantum circuit
            circuit_2: (Parametrized) quantum circuit

        Raises:
            ValueError: when ``circuit_1`` and ``circuit_2`` don't have the same number of qubits.
        """

        if circuit_1 is not None and circuit_2 is not None:
            if circuit_1.num_qubits != circuit_2.num_qubits:
                raise ValueError(
                    f"The number of qubits for the left circuit ({circuit_1.num_qubits}) \
                        and right circuit ({circuit_2.num_qubits}) do not coincide."
                )

    @abstractmethod
    def _create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Implementation-dependent method to create a fidelity circuit
        from 2 circuit inputs.
        Args:
            circuit_1: (Parametrized) quantum circuit
            circuit_2: (Parametrized) quantum circuit

        Returns:
            The fidelity quantum circuit corresponding to circuit_1 and circuit_2.
        """
        raise NotImplementedError

    def _set_circuits(
        self,
        circuits_1: Sequence[QuantumCircuit] | None = None,
        circuits_2: Sequence[QuantumCircuit] | None = None,
    ) -> None:
        """
        Update the list of fidelity circuits to be evaluated.
        These circuits represent the state overlap between pairs of input circuits,
        and their construction depends on the fidelity method implementations.

        Args:
            circuits_1: (Parametrized) quantum circuits.
            circuits_2: (Parametrized) quantum circuits.

        Raises:
            ValueError: if the length of the input circuit lists doesn't match.
        """

        if not len(circuits_1) == len(circuits_2):
            raise ValueError(
                f"The length of the first circuit list({len(circuits_1)}) \
                    and second circuit list ({len(circuits_2)}) does not coincide."
            )

        circuits = []
        for (circuit_1, circuit_2) in zip(circuits_1, circuits_2):

            circuit = self._circuit_cache.get((id(circuit_1), id(circuit_2)))

            if circuit is not None:
                circuits.append(circuit)
            else:
                self._check_qubits_mismatch(circuit_1, circuit_2)

                # re-parametrize input circuits
                left_parameters = ParameterVector("x", circuit_1.num_parameters)
                self._left_parameters.append(left_parameters)
                parametrized_circuit_1 = circuit_1.assign_parameters(left_parameters)

                right_parameters = ParameterVector("y", circuit_2.num_parameters)
                self._right_parameters.append(right_parameters)
                parametrized_circuit_2 = circuit_2.assign_parameters(right_parameters)

                circuit = self._create_fidelity_circuit(
                    parametrized_circuit_1, parametrized_circuit_2
                )
                circuits.append(circuit)
                # update cache
                self._circuit_cache[id(circuit_1), id(circuit_2)] = circuit

        # set circuits
        self._circuits = circuits

    def _set_values(
        self, values_1: Sequence[Sequence[float]] | None, values_2: Sequence[Sequence[float]] | None
    ) -> None:
        """
        Update the list of parameter values to evaluate the corresponding
        fidelity circuits with.

        Args:
            values_1: Numerical parameters to be bound to the first circuits.
            values_2: Numerical parameters to be bound to the second circuits.

        Raises:
            ValueError: if the length of the input value lists doesn't match.
        """

        values = []
        if values_2 is not None or values_1 is not None:
            if values_2 is None:
                values = values_1
            elif values_1 is None:
                values = values_2
            else:
                for (val_1, val_2) in zip(values_1, values_2):
                    if len(val_1) != len(val_2):
                        raise ValueError(
                            f"The number of left parameters (currently {len(val_1)})"
                            f"has to be equal to the number of right parameters."
                            f"(currently {len(val_2)})"
                        )
                    values.append(val_1 + val_2)

        # set values
        self._parameter_values = values

    def _preprocess_inputs(
        self,
        circuits_1: Sequence[QuantumCircuit],
        circuits_2: Sequence[QuantumCircuit],
        values_1: Sequence[Sequence[float]] | None = None,
        values_2: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Preprocess input circuits and parameter values and update corresponding lists.

        Args:
           circuits_1: (Parametrized) quantum circuits preparing the first list of quantum states.
           circuits_2: (Parametrized) quantum circuits preparing the second list of quantum states.
           values_1: Numerical parameters to be bound to the first circuits.
           values_2: Numerical parameters to be bound to the second circuits.

        """

        self._set_circuits(circuits_1, circuits_2)

        values_1 = self._check_values_shape(values_1, circuits_1, "1")
        values_2 = self._check_values_shape(values_2, circuits_2, "2")

        self._set_values(values_1, values_2)

    @abstractmethod
    def evaluate(
        self,
        circuits_1: Sequence[QuantumCircuit],
        circuits_2: Sequence[QuantumCircuit],
        values_1: Sequence[Sequence[float]] | None = None,
        values_2: Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> list[float]:
        """Compute the state overlap (fidelity) calculation between 2
        parametrized circuits (first and second) for a specific set of parameter
        values (first and second). This calculation depends on the particular
        fidelity method implementation.
        Args:
            circuits_1: (Parametrized) quantum circuits preparing one set of states
            circuits_2: (Parametrized) quantum circuits preparing another set of states
            values_1: Numerical parameters to be bound to the first circuits
            values_2: Numerical parameters to be bound to the second circuits.
            run_options: Backend runtime options used for circuit execution.
        """
        ...
