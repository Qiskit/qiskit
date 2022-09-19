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
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import _make_lin_comb_gradient_circuit, _make_lin_comb_qfi_circuit
from .lin_comb_estimator_gradient import LinCombEstimatorGradient

Pauli_Z = Pauli("Z")


class QFI:
    def __init__(self, estimator: BaseEstimator, **run_options):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in ``run`` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """
        self._estimator = estimator
        self._gradient_circuits = {}
        self._gradient = LinCombEstimatorGradient(estimator)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        aux_meas_op=None,
        **run_options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        jobs, result_indices_all, coeffs_all, metadata_, phasefixes = [], [], [], [], []

        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            if parameters_ is None:
                param_set = set(circuit.parameters)
            else:
                param_set = set(parameters_)
            metadata_.append({"parameters": [p for p in circuit.parameters if p in param_set]})
            observable_ = observable.expand(Pauli_Z)

            gradient_result = self._gradient.run(
                circuits=[circuit],
                observables=[observable],
                parameter_values=[parameter_values_],
                aux_meas_op=True,
            ).result()

            phasefix = np.outer(
                np.conjugate(gradient_result.gradients[0]), gradient_result.gradients[0]
            )
            phasefixes.append(phasefix)
            qfi_circuits_ = _make_lin_comb_qfi_circuit(circuit)

            n = len(circuit.parameters)
            # only compute the gradients for parameters in the parameter set
            qfi_circuits, result_indices, coeffs = [], [], []
            # result_indices = np.array((len(param_set), len(param_set)), dtype=int)
            # coeffs = np.array((len(param_set), len(param_set)), dtype=float)
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

            n = len(qfi_circuits)
            job = self._estimator.run(
                qfi_circuits, [observable_] * n, [parameter_values_] * n, **run_options
            )
            jobs.append(job)
            result_indices_all.append(result_indices)
            coeffs_all.append(coeffs)

        # combine the results
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        qfis = []
        for i, result in enumerate(results):

            qfi_ = np.zeros((len(metadata_[i]["parameters"]), len(metadata_[i]["parameters"])))
            for grad_, idx, coeff in zip(result.values, result_indices_all[i], coeffs_all[i]):
                qfi_[idx] += coeff * grad_
            qfi = qfi_ - phasefixes[i]
            qfi += np.triu(qfi_, k=1).T
            qfis.append(qfi)

        return EstimatorGradientResult(gradients=qfis, metadata=metadata_, run_options=run_options)

        #     # Make the observable as observable as :class:`~qiskit.quantum_info.SparsePauliOp`.
        #     observable = init_observable(observable)
        #     # a set of parameters to be differentiated
        #     if parameters_ is None:
        #         param_set = set(circuit.parameters)
        #     else:
        #         param_set = set(parameters_)
        #     metadata_.append({"parameters": [p for p in circuit.parameters if p in param_set]})

        #     if aux_meas_op is not None:
        #         op2 = SparsePauliOp.from_list([("Z", 1), ("Y", complex(0, -1))])
        #         observable_ = observable.expand(op2)
        #     else:
        #         observable_ = observable.expand(Pauli_Z)
        #     gradient_circuits_ = self._gradient_circuits.get(id(circuit))
        #     if gradient_circuits_ is None:
        #         gradient_circuits_ = _make_lin_comb_gradient_circuit(circuit)
        #         self._gradient_circuits[id(circuit)] = gradient_circuits_

        #     # only compute the gradients for parameters in the parameter set
        #     gradient_circuits, result_indices, coeffs = [], [], []
        #     result_idx = 0
        #     for i, param in enumerate(circuit.parameters):
        #         if param in param_set:
        #             gradient_circuits.extend(
        #                 grad.gradient_circuit for grad in gradient_circuits_[param]
        #             )

        #             result_indices.extend(result_idx for _ in gradient_circuits_[param])
        #             result_idx += 1
        #             for grad_data in gradient_circuits_[param]:
        #                 coeff = grad_data.coeff
        #                 # if the parameter is a parameter expression, we need to substitute
        #                 if isinstance(coeff, ParameterExpression):
        #                     local_map = {
        #                         p: parameter_values_[circuit.parameters.data.index(p)]
        #                         for p in coeff.parameters
        #                     }
        #                     bound_coeff = float(coeff.bind(local_map))
        #                 else:
        #                     bound_coeff = coeff
        #                 coeffs.append(bound_coeff)

        #     n = len(gradient_circuits)
        #     job = self._estimator.run(
        #         gradient_circuits, [observable_] * n, [parameter_values_] * n, **run_options
        #     )
        #     jobs.append(job)
        #     result_indices_all.append(result_indices)
        #     coeffs_all.append(coeffs)

        # # combine the results
        # try:
        #     results = [job.result() for job in jobs]
        # except Exception as exc:
        #     raise AlgorithmError("Estimator job failed.") from exc

        # gradients = []
        # for i, result in enumerate(results):
        #     gradient_ = np.zeros(len(metadata_[i]["parameters"]), dtype="complex_")
        #     for grad_, idx, coeff in zip(result.values, result_indices_all[i], coeffs_all[i]):
        #         gradient_[idx] += coeff * grad_
        #     gradients.append(gradient_)

        # # TODO: include primitive's run_options as well
        # return EstimatorGradientResult(
        #     gradients=gradients, metadata=metadata_, run_options=run_options
        # )
