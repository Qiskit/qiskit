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
from .derive_circuit import derive_circuit
from .split_circuits import split

from ..base.base_estimator_gradient import BaseEstimatorGradient
from ..base.estimator_gradient_result import EstimatorGradientResult
from ..utils import DerivativeType

logger = logging.getLogger(__name__)


class ReverseEstimatorGradient(BaseEstimatorGradient):
    """Estimator gradients with the classically efficient reverse mode.

    .. note::

        This gradient implementation is based on statevector manipulations and scales
        exponentially with the number of qubits. However, for small system sizes it can be very fast
        compared to circuit-based gradients.

    This class implements the calculation of the expectation gradient as described in
    [1]. By keeping track of two statevectors and iteratively sweeping through each parameterized
    gate, this method scales only linearly with the number of parameters.

    **References:**

        [1]: Jones, T. and Gacon, J. "Efficient calculation of gradients in classical simulations
             of variational quantum algorithms" (2020).
             `arXiv:2009.02823 <https://arxiv.org/abs/2009.02823>`_.

    """

    SUPPORTED_GATES = ["rx", "ry", "rz", "cp", "crx", "cry", "crz"]

    def __init__(self, derivative_type: DerivativeType = DerivativeType.REAL):
        """
        Args:
            derivative_type: Defines whether the real, imaginary or real plus imaginary part
                of the gradient is returned.
        """
        dummy_estimator = Estimator()  # this is required by the base class, but not used
        super().__init__(dummy_estimator, derivative_type=derivative_type)

    @BaseEstimatorGradient.derivative_type.setter
    def derivative_type(self, derivative_type: DerivativeType) -> None:
        """Set the derivative type."""
        self._derivative_type = derivative_type

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the gradients of the expectation values by the parameter shift rule."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, **options
        )
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,  # pylint: disable=unused-argument
    ) -> EstimatorGradientResult:
        num_gradients = len(circuits)
        gradients = []
        metadata = []

        for i in range(num_gradients):
            # temporary variables for easier access
            circuit = circuits[i]
            parameters_ = parameters[i]
            observable = observables[i]
            values = parameter_values[i]

            # the metadata only contains the parameters as there are no run configs here
            metadata.append(
                {
                    "parameters": parameters_,
                    "derivative_type": self.derivative_type,
                }
            )

            # keep track of the parameter order of the circuit, as the circuit splitting might
            # produce a list of unitaries in a different order
            # original_parameter_order = [p for p in circuit.parameters if p in parameters_]

            # split the circuit and generate lists of unitaries [U_1, U_2, ...] and
            # parameters [p_1, p_2, ...] in these unitaries
            unitaries, paramlist = split(circuit, parameters=parameters_)

            parameter_binds = dict(zip(circuit.parameters, values))
            bound_circuit = bind(circuit, parameter_binds)

            # initialize state variables -- we use the same naming as in the paper
            phi = Statevector(bound_circuit)
            lam = _evolve_by_operator(observable, phi)

            # store gradients in a dictionary to return them in the correct order
            grads = {param: 0j for param in parameters_}

            num_parameters = len(unitaries)
            for j in reversed(range(num_parameters)):
                unitary_j = unitaries[j]

                # We currently only support gates with a single parameter -- which is reflected
                # in self.SUPPORTED_GATES -- but generally we could also support gates with multiple
                # parameters per gate
                parameter_j = paramlist[j][0]

                # get the analytic gradient d U_j / d p_j and bind the gate
                deriv = derive_circuit(unitary_j, parameter_j)
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
        # this disable is needed as Pylint does not understand derivative_type is a property if
        # it is only defined in the base class and the getter is in the child
        # pylint: disable=comparison-with-callable
        if self.derivative_type == DerivativeType.REAL:
            return 2 * np.real(gradient)
        if self.derivative_type == DerivativeType.IMAG:
            return 2 * np.imag(gradient)

        return 2 * gradient


def _evolve_by_operator(operator, state):
    """Evolve the Statevector state by operator."""

    # try casting to sparse matrix and use sparse matrix-vector multiplication, which is
    # a lot faster than using Statevector.evolve
    if isinstance(operator, PauliSumOp):
        operator = operator.primitive * operator.coeff

    try:
        spmatrix = operator.to_matrix(sparse=True)
        evolved = spmatrix @ state.data
        return Statevector(evolved)
    except (TypeError, AttributeError):
        logger.info("Operator is not castable to a sparse matrix, using Statevector.evolve.")

    return state.evolve(operator)
