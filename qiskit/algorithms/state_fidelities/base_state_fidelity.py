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
Base state fidelity interface
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence, Mapping
import numpy as np

from qiskit import QuantumCircuit
from qiskit.algorithms import AlgorithmJob
from qiskit.circuit import ParameterVector
from .state_fidelity_result import StateFidelityResult


class BaseStateFidelity(ABC):
    """
    An interface to calculate state_fidelities (state overlaps) for pairs of
    (parametrized) quantum circuits.
    """

    def __init__(
        self,
    ) -> None:
        """Initializes the class to evaluate state fidelities."""

        # use cache for preventing unnecessary circuit compositions
        self._circuit_cache: Mapping[(int, int), QuantumCircuit] = {}

    @staticmethod
    def _preprocess_values(
        circuits: QuantumCircuit,
        values: Sequence[Sequence[float]] | None = None,
    ) -> Sequence[Sequence[float]] | None:
        """
        Checks whether the passed values match the shape of the parameters
        of the corresponding circuits and formats values to 2D list.

        Args:
            circuits: list of circuits to be checked
            values: parameter values corresponding to the circuits to be checked

        Returns:
            Returns a 2D value list if values match, and an empty 2D list
            if no parameters are passed.

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
                        f"`values` cannot be `None` because circuit <{circuit.name}> has "
                        f"{circuit.num_parameters} free parameters."
                    )
            return [[]]
        else:
            # ensure 2d list
            if not isinstance(values, list):
                values = [values]
            if len(values) > 0 and not isinstance(values[0], list):
                values = [values]
            return values

    def _check_qubits_match(self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit) -> None:
        """
        Checks that the number of qubits of 2 circuits matches.
        Args:
            circuit_1: (Parametrized) quantum circuit
            circuit_2: (Parametrized) quantum circuit

        Raises:
            ValueError: when ``circuit_1`` and ``circuit_2`` don't have the
            same number of qubits.
        """

        if circuit_1.num_qubits != circuit_2.num_qubits:
            raise ValueError(
                f"The number of qubits for the first circuit ({circuit_1.num_qubits}) "
                f"and second circuit ({circuit_2.num_qubits}) do not coincide."
            )

    @abstractmethod
    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Implementation-dependent method to create a fidelity circuit
        from 2 circuit inputs.
        Args:
            circuit_1: (Parametrized) quantum circuit
            circuit_2: (Parametrized) quantum circuit

        Returns:
            The fidelity quantum circuit corresponding to ``circuit_1`` and ``circuit_2``.
        """
        raise NotImplementedError

    def _construct_circuits(
        self,
        circuits_1: Sequence[QuantumCircuit],
        circuits_2: Sequence[QuantumCircuit],
    ) -> Sequence[QuantumCircuit]:
        """
        Construct the list of fidelity circuits to be evaluated.
        These circuits represent the state overlap between pairs of input circuits,
        and their construction depends on the fidelity method implementations.

        Args:
            circuits_1: (Parametrized) quantum circuits.
            circuits_2: (Parametrized) quantum circuits.

        Returns:
            List of constructed fidelity circuits.

        Raises:
            ValueError: if the length of the input circuit lists doesn't match.
        """

        if not len(circuits_1) == len(circuits_2):
            raise ValueError(
                f"The length of the first circuit list({len(circuits_1)}) "
                f"and second circuit list ({len(circuits_2)}) is not the same."
            )

        circuits = []
        for (circuit_1, circuit_2) in zip(circuits_1, circuits_2):

            # TODO: improve caching, what if the circuit is modified without changing the id?
            circuit = self._circuit_cache.get((id(circuit_1), id(circuit_2)))

            if circuit is not None:
                circuits.append(circuit)
            else:
                self._check_qubits_match(circuit_1, circuit_2)

                # re-parametrize input circuits
                # TODO: make smarter checks to avoid unnecesary reparametrizations
                parameters_1 = ParameterVector("x", circuit_1.num_parameters)
                parametrized_circuit_1 = circuit_1.assign_parameters(parameters_1)
                parameters_2 = ParameterVector("y", circuit_2.num_parameters)
                parametrized_circuit_2 = circuit_2.assign_parameters(parameters_2)

                circuit = self.create_fidelity_circuit(
                    parametrized_circuit_1, parametrized_circuit_2
                )
                circuits.append(circuit)
                # update cache
                self._circuit_cache[id(circuit_1), id(circuit_2)] = circuit

        return circuits

    def _construct_value_list(
        self,
        circuits_1: Sequence[QuantumCircuit],
        circuits_2: Sequence[QuantumCircuit],
        values_1: Sequence[Sequence[float]] | None = None,
        values_2: Sequence[Sequence[float]] | None = None,
    ) -> Sequence[float]:
        """
        Preprocess input parameter values to match the fidelity
        circuit parametrization, and return in list format.

        Args:
           circuits_1: (Parametrized) quantum circuits preparing the
                        first list of quantum states.
           circuits_2: (Parametrized) quantum circuits preparing the
                        second list of quantum states.
           values_1: Numerical parameters to be bound to the first circuits.
           values_2: Numerical parameters to be bound to the second circuits.

        Returns:
             List of parameter values for fidelity circuit.

        """
        values_1 = self._preprocess_values(circuits_1, values_1)
        values_2 = self._preprocess_values(circuits_2, values_2)

        values = []
        if len(values_2[0]) == 0:
            values = values_1
        elif len(values_1[0]) == 0:
            values = values_2
        else:
            for (val_1, val_2) in zip(values_1, values_2):
                values.append(val_1 + val_2)

        return values

    @abstractmethod
    def _run(
        self,
        circuits_1: Sequence[QuantumCircuit],
        circuits_2: Sequence[QuantumCircuit],
        values_1: Sequence[Sequence[float]] | None = None,
        values_2: Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> StateFidelityResult:
        r"""
        Compute the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second). This calculation depends on the particular
        fidelity method implementation, but always represents:

            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`

        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``circuit_1`` and ``circuit_2``, respectively.

        Args:
            circuits_1: (Parametrized) quantum circuits preparing :math:`|\psi\rangle`.
            circuits_2: (Parametrized) quantum circuits preparing :math:`|\phi\rangle`.
            values_1: Numerical parameters to be bound to the first set of circuits
            values_2: Numerical parameters to be bound to the second set of circuits.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The result of the fidelity calculation.
        """
        raise NotImplementedError

    def run(
        self,
        circuits_1: Sequence[QuantumCircuit],
        circuits_2: Sequence[QuantumCircuit],
        values_1: Sequence[Sequence[float]] | None = None,
        values_2: Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> AlgorithmJob:
        r"""
        Run asynchronously the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second). This calculation depends on the particular
        fidelity method implementation, but always represents:

        .. math::

            |\langle\psi(x)|\phi(y)\rangle|^2

        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``circuit_1`` and ``circuit_2``, respectively.

        Args:
            circuits_1: (Parametrized) quantum circuits preparing :math:`|\psi\rangle`.
            circuits_2: (Parametrized) quantum circuits preparing :math:`|\phi\rangle`.
            values_1: Numerical parameters to be bound to the first set of circuits.
            values_2: Numerical parameters to be bound to the second set of circuits.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            Primitive job for the fidelity calculation.
            The job's result is an instance of ``StateFidelityResult``.
        """

        job = AlgorithmJob(self._run, circuits_1, circuits_2, values_1, values_2, **run_options)

        job.submit()
        return job

    def _truncate_fidelities(self, fidelities: Sequence[float]) -> Sequence[float]:
        """
        Ensure fidelity result in [0,1]

        Args:
           fidelities: sequence of raw fidelity results

        Returns:
             List of truncated fidelities.

        """
        return np.clip(fidelities, 0, 1).tolist()
