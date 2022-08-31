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
Compute-uncompute fidelity interface using primitives
"""

from __future__ import annotations
from collections.abc import Sequence
from copy import copy

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler
from .base_state_fidelity import BaseStateFidelity
from .state_fidelity_result import StateFidelityResult


class ComputeUncompute(BaseStateFidelity):
    """
    This class leverages the sampler primitive to calculate the fidelity of
    two quantum circuits with the compute-uncompute method.
    """

    def __init__(self, sampler: BaseSampler, **run_options) -> None:
        r"""
        Initializes the class to evaluate the compute-uncompute state fidelity using
        the sampler primitive. This fidelity is defined as the state overlap

            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`,

        where :math:`x` and :math:`y` are optional parametrizations of the
        states :math:`\psi` and :math:`\phi` prepared by the circuits
        ``circuit_1`` and ``circuit_2``, respectively.
        Args:
            sampler: Sampler primitive instance.
            run_options: Backend runtime options used for circuit execution.
        """
        self._sampler = sampler
        self._default_run_options = run_options
        super().__init__()

    def create_fidelity_circuit(self, circuit_1, circuit_2) -> QuantumCircuit:
        """
        Creates fidelity circuit following the compute-uncompute method.
        Args:
            circuit_1: (Parametrized) quantum circuit
            circuit_2: (Parametrized) quantum circuit

        Returns:
            The fidelity quantum circuit corresponding to circuit_1 and circuit_2.
        """
        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        return circuit

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
        values (first and second) following the compute-uncompute method

        The fidelity corresponds to:

            :math:`|\langle\psi(x)|\phi(y)\rangle|^2`

        Args:
            circuits_1: (Parametrized) quantum circuits preparing :math:`|\psi\rangle`.
            circuits_2: (Parametrized) quantum circuits preparing :math:`|\phi\rangle`.
            values_1: Numerical parameters to be bound to the first circuits.
            values_2: Numerical parameters to be bound to the second circuits.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The result of the fidelity calculation.

        Raises:
            ValueError: At least one pair of circuits must be defined.
        """

        circuits = self._construct_circuits(circuits_1, circuits_2)
        values = self._construct_value_list(circuits_1, circuits_2, values_1, values_2)

        if len(circuits) == 0:
            raise ValueError(
                "At least one pair of circuits must be defined to calculate the state overlap."
            )

        # The priority of run options is as follows:
        # run_options in `evaluate` method > fidelity's default run_options >
        # primitive's default run_options.
        run_opts = copy(self._default_run_options)
        run_opts.update(**run_options)

        if len(values) > 0:
            job = self._sampler.run(circuits=circuits, parameter_values=values, **run_opts)
        else:
            job = self._sampler.run(circuits=circuits, **run_opts)

        result = job.result()

        raw_fidelities = [prob_dist.get(0, 0) for prob_dist in result.quasi_dists]
        fidelities = self._truncate_fidelities(raw_fidelities)

        return StateFidelityResult(
            fidelities=fidelities, raw_fidelities=raw_fidelities, metadata=run_opts
        )
