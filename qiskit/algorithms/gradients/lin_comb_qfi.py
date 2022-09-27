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
A class for the Quantum Fisher Information.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import init_observable
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_qfi import BaseQFI
from .lin_comb_estimator_gradient import DerivativeType, LinCombEstimatorGradient
from .qfi_result import QFIResult
from .utils import _make_lin_comb_qfi_circuit


class LinCombQFI(BaseQFI):
    """Computes the Quantum Fisher Information (QFI) given a pure,
    parameterized quantum state. This method employs a linear
    combination of unitaries [1].

    **Reference:**
    [1] Schuld et al., Evaluating analytic gradients on quantum hardware, 2018
    `arXiv:1811.11184 <https://arxiv.org/pdf/1811.11184.pdf>`_

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        phase_fix: bool = True,
        derivative_type: DerivativeType = DerivativeType.REAL,
        **run_options,
    ):
        """
        Args:
            estimator: The estimator used to compute the QFI.
            phase_fix: Whether to calculate the second term (phase fix) of the QFI, which is
                Re[(dω⟨<ψ(ω)|)|ψ(ω)><ψ(ω)|(dω|ψ(ω))>]. Default to ``True``.
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``. Defaults to
                ``DerivativeType.REAL``.
                For ``DerivativeType.REAL`` we compute 4Re[(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉],
                for ``DerivativeType.IMAG`` we compute 4Im[(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉], and
                for ``DerivativeType.COMPLEX`` we compute 4(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉.

            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in ``run`` method > QFI's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """
        super().__init__(estimator, **run_options)
        self._phase_fix = phase_fix
        self._derivative_type = derivative_type
        self._gradient = LinCombEstimatorGradient(
            estimator, derivative_type=DerivativeType.COMPLEX, **run_options
        )
        self._qfi_circuits = {}

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[complex]],
        parameters: Sequence[Sequence[Parameter] | None],
        **run_options,
    ) -> QFIResult:
        """Compute the estimator QFIs on the given circuits."""
        jobs, result_indices_all, coeffs_all, metadata_, gradient_jobs, phase_fixes = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # Make the observable as observable as :class:`~qiskit.quantum_info.SparsePauliOp`.
            observable = init_observable(observable)
            # a set of parameters to be differentiated
            if parameters_ is None:
                param_set = set(circuit.parameters)
            else:
                param_set = set(parameters_)
            metadata_.append({"parameters": [p for p in circuit.parameters if p in param_set]})

            # compute the second term (the phase fix) in the QFI
            if self._phase_fix:
                gradient_job = self._gradient.run(
                    circuits=[circuit],
                    observables=[observable],
                    parameter_values=[parameter_values_],
                    parameters=[parameters_],
                    **run_options,
                )
                gradient_jobs.append(gradient_job)

            # compute the first term in the QFI
            qfi_circuits_ = self._qfi_circuits.get(id(circuit))
            if qfi_circuits_ is None:
                qfi_circuits_ = _make_lin_comb_qfi_circuit(circuit)
                self._qfi_circuits[id(circuit)] = qfi_circuits_

            # only compute the gradients for parameters in the parameter set
            qfi_circuits, result_indices, coeffs = [], [], []
            result_map = {}
            idx = 0
            for i, param in enumerate(circuit.parameters):
                if param in param_set:
                    result_map[i] = idx
                    idx += 1
                else:
                    result_map[i] = -1

            result_indices = []
            for i, param_i in enumerate(circuit.parameters):
                if not param_i in param_set:
                    continue
                for j, param_j in enumerate(circuit.parameters):
                    if not param_j in param_set or i > j:
                        continue
                    qfi_circuits.extend(grad.gradient_circuit for grad in qfi_circuits_[i, j])
                    result_indices.extend(
                        (result_map[i], result_map[j]) for _ in qfi_circuits_[i, j]
                    )
                    for grad in qfi_circuits_[i, j]:
                        coeff = grad.coeff
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

            observable_ = self._expand_observable(observable)

            n = len(qfi_circuits)
            job = self._estimator.run(
                qfi_circuits, [observable_] * n, [parameter_values_] * n, **run_options
            )
            jobs.append(job)
            result_indices_all.append(result_indices)
            coeffs_all.append(coeffs)

        # combine the results
        try:
            gradient_results = [g_job.result() for g_job in gradient_jobs]
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        # compute the phase fix
        if self._phase_fix:
            for gradient_result in gradient_results:
                phase_fix_ = np.outer(
                    np.conjugate(gradient_result.gradients[0]), gradient_result.gradients[0]
                )
                phase_fixes.append(phase_fix_)
        else:
            phase_fixes = [
                np.zeros((len(metadata_[i]["parameters"]), len(metadata_[i]["parameters"])))
                for i in range(len(circuits))
            ]

        qfis = []
        for i, result in enumerate(results):
            qfi_ = np.zeros(
                (len(metadata_[i]["parameters"]), len(metadata_[i]["parameters"])), dtype="complex_"
            )
            for grad_, idx, coeff in zip(result.values, result_indices_all[i], coeffs_all[i]):
                qfi_[idx] += coeff * grad_
            qfi = qfi_ - phase_fixes[i]
            qfi += np.triu(qfi_, k=1).T
            qfis.append(qfi)

        run_opt = self._get_local_run_options(run_options)
        return QFIResult(qfis=qfis, metadata=metadata_, run_options=run_opt)

    def _expand_observable(self, observable: BaseOperator | PauliSumOp) -> BaseOperator:
        """Expands the observable based on the derivative type."""
        if self._derivative_type == DerivativeType.REAL:
            op2 = SparsePauliOp.from_list([("Z", 1)])
        elif self._derivative_type == DerivativeType.IMAG:
            op2 = SparsePauliOp.from_list([("Y", -1)])
        elif self._derivative_type == DerivativeType.COMPLEX:
            op2 = SparsePauliOp.from_list([("Z", 1), ("Y", complex(0, -1))])
        else:
            raise ValueError(f"Derivative type {self._derivative_type} is not supported.")
        return observable.expand(op2)
