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
Gradient of probabilities with linear combination of unitaries (LCU)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import init_observable
from qiskit.quantum_info import Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import _make_lin_comb_gradient_circuit


Pauli_Z = Pauli("Z")


class LinCombEstimatorGradient(BaseEstimatorGradient):
    """Compute the gradients of the expectation values.
    This method employs a linear combination of unitaries [1].

    **Reference:**
    [1] Schuld et al., Evaluating analytic gradients on quantum hardware, 2018
    `arXiv:1811.11184 <https://arxiv.org/pdf/1811.11184.pdf>`_
    """

    def __init__(self, estimator: BaseEstimator, **options):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
        """
        self._gradient_circuits = {}
        super().__init__(estimator, **options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        jobs, result_indices_all, coeffs_all, metadata_ = [], [], [], []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # Make the observable as :class:`~qiskit.quantum_info.SparsePauliOp`.
            observable = init_observable(observable)
            # a set of parameters to be differentiated
            if parameters_ is None:
                param_set = set(circuit.parameters)
            else:
                param_set = set(parameters_)
            metadata_.append({"parameters": [p for p in circuit.parameters if p in param_set]})

            # TODO: support measurement in different basis (Y and Z+iY)
            observable_ = observable.expand(Pauli_Z)
            gradient_circuits_ = self._gradient_circuits.get(id(circuit))
            if gradient_circuits_ is None:
                gradient_circuits_ = _make_lin_comb_gradient_circuit(circuit)
                self._gradient_circuits[id(circuit)] = gradient_circuits_

            # only compute the gradients for parameters in the parameter set
            gradient_circuits, result_indices, coeffs = [], [], []
            result_idx = 0
            for i, param in enumerate(circuit.parameters):
                if param in param_set:
                    gradient_circuits.extend(
                        grad.gradient_circuit for grad in gradient_circuits_[param]
                    )

                    result_indices.extend(result_idx for _ in gradient_circuits_[param])
                    result_idx += 1
                    for grad_data in gradient_circuits_[param]:
                        coeff = grad_data.coeff
                        # if the parameter is a parameter expression, we need to substitute
                        if isinstance(coeff, ParameterExpression):
                            local_map = {
                                p: parameter_values_[circuit.parameters.data.index(p)]
                                for p in coeff.parameters
                            }
                            bound_coeff = float(coeff.bind(local_map))
                        else:
                            bound_coeff = coeff
                        coeffs.append(bound_coeff)

            n = len(gradient_circuits)
            job = self._estimator.run(
                gradient_circuits, [observable_] * n, [parameter_values_] * n, **options
            )
            jobs.append(job)
            result_indices_all.append(result_indices)
            coeffs_all.append(coeffs)

        # combine the results
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        gradients = []
        for i, result in enumerate(results):
            gradient_ = np.zeros(len(metadata_[i]["parameters"]))
            for grad_, idx, coeff in zip(result.values, result_indices_all[i], coeffs_all[i]):
                gradient_[idx] += coeff * grad_
            gradients.append(gradient_)

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata_, options=opt)
