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
from qiskit.primitives.primitive_job import PrimitiveJob
from .base_fidelity import BaseFidelity
# from .fidelity_job import FidelityJob

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
        """
        Initializes the class to evaluate the fidelities defined as the state overlap
            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`,
        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``left_circuit`` and ``right_circuit``, respectively.
        Args:
            left_circuit: (Parametrized) quantum circuit :math:`|\psi\rangle`.
            right_circuit: (Parametrized) quantum circuit :math:`|\phi\rangle`.
            sampler: Sampler primitive instance.
        Raises:
            ValueError: left_circuit and right_circuit don't have the same number of qubits.
        """
        self.sampler = sampler
        super().__init__(left_circuit, right_circuit)

    def _call(
        self,
        left_circuit: Sequence[QuantumCircuit] | None = None,
        right_circuit: Sequence[QuantumCircuit] | None = None,
        left_values: Sequence[Sequence[float]] | None = None,
        right_values: Sequence[Sequence[float]] | None = None
    ) -> np.ndarray:
        """Run the state overlap (fidelity) calculation between 2
        parametrized circuits (left and right) for a specific set of parameter
        values (left and right).
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
            The result for the fidelity calculation.
        """
        if left_circuit is not None:
            self._set_circuits(left_circuit=left_circuit[0])
        if right_circuit is not None:
            self._set_circuits(right_circuit=right_circuit[0])

        if self._left_circuit is None or self._right_circuit is None:
            raise ValueError(
                "The left and right circuits must be defined to "
                "calculate the state overlap. "
            )

        circuit = self._left_circuit.compose(self._right_circuit.inverse())
        circuit.measure_all()
        self._circuit = circuit

        values_list = []
        for values, side in zip([left_values, right_values], ["left", "right"]):
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
            job = self.sampler.run(circuits=[self._circuit] * len(values), parameter_values=values)
        else:
            job = self.sampler.run(circuits=[self._circuit])

        result = job.result()

        # if error mitigation is added in the future, we will have to handle
        # negative values in some way (e.g. clipping to zero)
        overlaps = [prob_dist.get(0, 0) for prob_dist in result.quasi_dists]
        return np.array(overlaps)

    def _run(
        self,
        left_circuit: Sequence[QuantumCircuit] | None = None,
        right_circuit: Sequence[QuantumCircuit] | None = None,
        left_values: Sequence[Sequence[float]] | None = None,
        right_values: Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> PrimitiveJob:
        """Run the asynchronous job of the state overlap (fidelity) calculation between 2
        parametrized circuits (left and right) for a specific set of parameter
        values (left and right).
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

        job = PrimitiveJob(
            self._call, left_circuit, right_circuit, left_values, right_values, **run_options
        )
        job.submit()
        return job
