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
from enum import Enum

from typing import Sequence

import numpy as np

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import init_observable
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import _make_lin_comb_gradient_circuit


class DerivativeType(Enum):
    """Types of derivative."""

    REAL = "real"
    IMAG = "imag"
    COMPLEX = "complex"


class LinCombEstimatorGradient(BaseEstimatorGradient):
    """Compute the gradients of the expectation values.
    This method employs a linear combination of unitaries [1].

    **Reference:**
    [1] Schuld et al., Evaluating analytic gradients on quantum hardware, 2018
    `arXiv:1811.11184 <https://arxiv.org/pdf/1811.11184.pdf>`_
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        derivative_type: DerivativeType = DerivativeType.REAL,
        options: Options | None = None,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the gradients.
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``. Defaults to
                ``DerivativeType.REAL``.

                    - ``DerivativeType.REAL`` computes :math:`2 \mathrm{Re}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.IMAG`` computes :math:`2 \mathrm{Im}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.COMPLEX`` computes :math:`2 ⟨ψ(ω)|O(θ)|dω ψ(ω)〉`.

            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting.
        """
        self._gradient_circuits = {}
        self._derivative_type = derivative_type
        super().__init__(estimator, options)

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

            meta = {"parameters": [p for p in circuit.parameters if p in param_set]}
            meta["derivative_type"] = self._derivative_type
            metadata_.append(meta)

            # if derivative_type is DerivativeType.COMPLEX, sum the real and imaginary parts later
            if self._derivative_type == DerivativeType.REAL:
                op1 = SparsePauliOp.from_list([("Z", 1)])
            elif self._derivative_type == DerivativeType.IMAG:
                op1 = SparsePauliOp.from_list([("Y", -1)])
            elif self._derivative_type == DerivativeType.COMPLEX:
                op1 = SparsePauliOp.from_list([("Z", 1)])
                op2 = SparsePauliOp.from_list([("Y", -1)])
            else:
                raise ValueError(f"Derivative type {self._derivative_type} is not supported.")

            observable_1 = observable.expand(op1)

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
            if self._derivative_type == DerivativeType.COMPLEX:
                observable_2 = observable.expand(op2)
                job = self._estimator.run(
                    gradient_circuits * 2,
                    [observable_1] * n + [observable_2] * n,
                    [parameter_values_] * 2 * n,
                    **options,
                )
                jobs.append(job)
            else:
                job = self._estimator.run(
                    gradient_circuits, [observable_1] * n, [parameter_values_] * n, **options
                )
                jobs.append(job)

            result_indices_all.append(result_indices)
            coeffs_all.append(coeffs)

        # combine the results
        try:
            results = [job.result() for job in jobs]
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        gradients = []
        for i, result in enumerate(results):
            gradient_ = np.zeros(len(metadata_[i]["parameters"]), dtype="complex")

            if metadata_[i]["derivative_type"] == DerivativeType.COMPLEX:
                n = len(result.values) // 2  # is always a multiple of 2
                for grad_, idx, coeff in zip(
                    result.values[:n], result_indices_all[i], coeffs_all[i]
                ):
                    gradient_[idx] += coeff * grad_
                for grad_, idx, coeff in zip(
                    result.values[n:], result_indices_all[i], coeffs_all[i]
                ):
                    gradient_[idx] += complex(0, coeff * grad_)
            else:
                for grad_, idx, coeff in zip(result.values, result_indices_all[i], coeffs_all[i]):
                    gradient_[idx] += coeff * grad_
                gradient_ = np.real(gradient_)
            gradients.append(gradient_)

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata_, options=opt)
