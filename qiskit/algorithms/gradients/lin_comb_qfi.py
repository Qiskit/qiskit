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
from copy import copy

import numpy as np

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp

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
        options: Options | None = None,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the QFI.
            phase_fix: Whether to calculate the second term (phase fix) of the QFI, which is
                :math:`\langle\partial_k \psi | \psi \rangle \langle\psi | \partial_l \psi \rangle`.
                Default to ``True``.
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``. Defaults to
                ``DerivativeType.REAL``.

                - ``DerivativeType.REAL`` computes

                .. math::

                    \mathrm{QFI}_{kl}= 4 \mathrm{Re}[\langle \partial_k \psi | \partial_l \psi \rangle
                        - \langle\partial_k \psi | \psi \rangle \langle\psi | \partial_l \psi \rangle].

                - ``DerivativeType.IMAG`` computes

                .. math::

                    \mathrm{QFI}_{kl}= 4 \mathrm{Im}[\langle \partial_k \psi | \partial_l \psi \rangle
                        - \langle\partial_k \psi | \psi \rangle \langle\psi | \partial_l \psi \rangle].

                - ``DerivativeType.COMPLEX`` computes

                .. math::

                    \mathrm{QFI}_{kl}= 4 [\langle \partial_k \psi | \partial_l \psi \rangle
                        - \langle\partial_k \psi | \psi \rangle \langle\psi | \partial_l \psi \rangle].

            options: Backend runtime options used for circuit execution. The order of priority is:
                options in ``run`` method > QFI's default options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """
        super().__init__(options)
        self._estimator: BaseEstimator = estimator
        self._phase_fix = phase_fix
        self._derivative_type = derivative_type
        self._gradient = LinCombEstimatorGradient(
            estimator, derivative_type=DerivativeType.COMPLEX, options=options
        )
        self._qfi_circuit_cache = {}

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> QFIResult:
        """Compute the QFIs on the given circuits."""
        jobs, result_indices_all, coeffs_all = [], [], []
        metadata_, gradient_jobs, phase_fixes = [], [], []

        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # a set of parameters to be differentiated
            result_map = {}
            idx = 0
            if parameters_ is None:
                param_set = set(circuit.parameters)
                result_map = {idx: idx for idx, _ in enumerate(circuit.parameters)}
            else:
                param_set = set(parameters_)
                for i, param in enumerate(circuit.parameters):
                    if param in param_set:
                        result_map[i] = idx
                        idx += 1
                    else:
                        result_map[i] = -1

            meta = {"parameters": [p for p in circuit.parameters if p in param_set]}
            meta["derivative_type"] = self._derivative_type
            metadata_.append(meta)

            observable = SparsePauliOp.from_list([("I" * circuit.num_qubits, 1)])
            # compute the second term (the phase fix) in the QFI
            if self._phase_fix:
                gradient_job = self._gradient.run(
                    circuits=[circuit],
                    observables=[observable],
                    parameter_values=[parameter_values_],
                    parameters=[parameters_],
                    **options,
                )
                gradient_jobs.append(gradient_job)
            # compute the first term in the QFI
            qfi_circuits_ = self._qfi_circuit_cache.get(_circuit_key(circuit))
            if qfi_circuits_ is None:
                # generate the all of the circuits for the first term in the QFI and cache them.
                # only the circuit related to specified parameters will be executed.
                # In the future, we can generate the specified circuits on demand.
                qfi_circuits_ = _make_lin_comb_qfi_circuit(circuit)
                self._qfi_circuit_cache[_circuit_key(circuit)] = qfi_circuits_

            # only compute the gradients for parameters in the parameter set
            qfi_circuits, result_indices, coeffs = [], [], []

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

            n = len(qfi_circuits)
            if self._derivative_type == DerivativeType.COMPLEX:
                observable_2 = observable.expand(op2)
                job = self._estimator.run(
                    qfi_circuits * 2,
                    [observable_1] * n + [observable_2] * n,
                    [parameter_values_] * 2 * n,
                    **options,
                )
                jobs.append(job)
            else:
                job = self._estimator.run(
                    qfi_circuits, [observable_1] * n, [parameter_values_] * n, **options
                )
                jobs.append(job)
            result_indices_all.append(result_indices)
            coeffs_all.append(coeffs)

        # combine the results
        try:
            gradient_results = [g_job.result() for g_job in gradient_jobs]
            results = [job.result() for job in jobs]
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job or gradient job failed.") from exc

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
                (len(metadata_[i]["parameters"]), len(metadata_[i]["parameters"])), dtype="complex"
            )

            if metadata_[i]["derivative_type"] == DerivativeType.COMPLEX:
                n = len(result.values) // 2  # is always a multiple of 2
                for grad_, idx, coeff in zip(
                    result.values[:n], result_indices_all[i], coeffs_all[i]
                ):
                    qfi_[idx] += coeff * grad_
                for grad_, idx, coeff in zip(
                    result.values[n:], result_indices_all[i], coeffs_all[i]
                ):
                    qfi_[idx] += complex(0, coeff * grad_)
            else:
                for grad_, idx, coeff in zip(result.values, result_indices_all[i], coeffs_all[i]):
                    qfi_[idx] += coeff * grad_
                qfi_ = qfi_.real

            if metadata_[i]["derivative_type"] == DerivativeType.REAL:
                phase_fixes[i] = phase_fixes[i].real
            elif metadata_[i]["derivative_type"] == DerivativeType.IMAG:
                phase_fixes[i] = phase_fixes[i].imag
            qfi = qfi_ - phase_fixes[i]
            qfi += np.triu(qfi_, k=1).T
            qfis.append(qfi)

        run_opt = self._get_local_options(options)
        return QFIResult(qfis=qfis, metadata=metadata_, options=run_opt)

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and QFI default options,
        where, if the same field is set in both, the QFI's default options override
        the primitive's default setting.

        Returns:
            The QFI default + estimator options.
        """
        opts = copy(self._estimator.options)
        opts.update_options(**self._default_options.__dict__)
        return opts

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the QFI default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > QFI's default options > primitive's
        default setting.

        Args:
            options: The fields to update the options

        Returns:
            The QFI default + estimator + run options.
        """
        opts = copy(self._estimator.options)
        opts.update_options(**options)
        return opts
