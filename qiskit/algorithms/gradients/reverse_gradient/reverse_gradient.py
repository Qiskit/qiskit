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

"""Estimator gradients with the classically efficient reverse mode."""

from __future__ import annotations
from collections.abc import Sequence
import logging

import numpy as np


from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator

from .bind import bind
from .gradient_lookup import analytic_gradient
from .split_circuits import split

from ..base_estimator_gradient import BaseEstimatorGradient
from ..estimator_gradient_result import EstimatorGradientResult
from ..lin_comb_estimator_gradient import DerivativeType

logger = logging.getLogger(__name__)


class ReverseEstimatorGradient(BaseEstimatorGradient):
    """Estimator gradients with the classically efficient reverse mode."""

    SUPPORTED_GATES = ["rx", "ry", "rz", "cp", "crx", "cry", "crz"]

    def __init__(self, derivative_type: DerivativeType = DerivativeType.REAL):
        """

        .. note::

            These gradients are calculated directly on the statevectors. This is
            inefficient in the number of qubits, but very fast in the number of gates.

        Args:
            derivative_type: Selects whether the real, imaginary or real + imaginary part
                of the gradient is returned.

        """
        alibi_estimator = Estimator()  # this is never used
        super().__init__(alibi_estimator)
        self.derivative_type = derivative_type

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the gradients of the expectation values by the parameter shift rule."""
        g_circuits, g_parameter_values, g_parameter_sets = self._preprocess(
            circuits, parameter_values, parameter_sets, self.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameter_sets, **options
        )
        return self._postprocess(results, circuits, parameter_values, parameter_sets)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        **options,  # pylint: disable=unused-argument
    ) -> EstimatorGradientResult:
        num_gradients = len(circuits)
        gradients = []
        metadata = []

        for i in range(num_gradients):
            # temporary variables for easier access
            circuit = circuits[i]
            parameters = parameter_sets[i]
            observable = observables[i]
            values = parameter_values[i]

            # the metadata only contains the parameters as there are no run configs here
            metadata.append(
                {"parameters": [p for p in circuits[i].parameters if p in parameter_sets[i]]}
            )

            # keep track of the parameter order of the circuit, as the circuit splitting might
            # produce a list of unitaries in a different order
            original_parameter_order = [p for p in circuit.parameters if p in parameters]

            # split the circuit and generate lists of unitaries [U_1, U_2, ...] and
            # parameters [p_1, p_2, ...] in these unitaries
            unitaries, paramlist = split(circuit, parameters=parameters)

            parameter_binds = dict(zip(circuit.parameters, values))
            bound_circuit = bind(circuit, parameter_binds)

            # initialize state variables -- we use the same naming as in the paper
            phi = Statevector(bound_circuit)
            lam = evolve_by_operator(observable, phi)

            # store gradients in a dictionary to return them in the correct order
            grads = {param: 0j for param in original_parameter_order}

            num_parameters = len(unitaries)
            for j in reversed(range(num_parameters)):
                unitary_j = unitaries[j]

                # We currently only support gates with a single parameter -- which is reflected
                # in self.SUPPORTED_GATES -- but generally we could also support gates with multiple
                # parameters per gate
                parameter_j = paramlist[j][0]

                # get the analytic gradient d U_j / d p_j and bind the gate
                deriv = analytic_gradient(unitary_j, parameter_j)
                for _, gate in deriv:
                    bind(gate, parameter_binds, inplace=True)

                # iterate the state variable
                unitary_j_dagger = bind(unitary_j, parameter_binds).inverse()
                phi = phi.evolve(unitary_j_dagger)

                # compute current gradient
                grad = sum(
                    coeff * lam.conjugate().data.dot(phi.evolve(gate).data) for coeff, gate in deriv
                )

                # Compute the full gradient (real and complex parts) as all information is available.
                # Later, based on the derivative type, cast to real/imag/complex.
                grads[parameter_j] += grad

                if j > 0:
                    lam = lam.evolve(unitary_j_dagger)

            gradient = np.array(list(grads.values()))
            gradients.append(self._to_derivtype(gradient))

        result = EstimatorGradientResult(gradients, metadata=metadata, options={})
        return result

    def _to_derivtype(self, gradient):
        if self.derivative_type == DerivativeType.REAL:
            return 2 * np.real(gradient)
        if self.derivative_type == DerivativeType.IMAG:
            return 2 * np.imag(gradient)
        return 2 * gradient


def evolve_by_operator(operator, state):
    """Evolve the Statevector state by operator."""

    # try casting to sparse matrix and use sparse matrix-vector multiplication, which is
    # a lot faster than using Statevector.evolve
    if isinstance(operator, PauliSumOp):
        operator = operator.primitive * operator.coeff

    try:
        spmatrix = operator.to_matrix(sparse=True)
        evolved = spmatrix @ state.data
        return Statevector(evolved)
    except AttributeError:
        logger.info("Operator is not castable to a sparse matrix, using Statevector.evolve.")

    return state.evolve(operator)
