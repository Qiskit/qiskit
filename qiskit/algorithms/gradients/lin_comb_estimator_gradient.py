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
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import init_observable, _circuit_key
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult
from .utils import DerivativeType, _make_lin_comb_gradient_circuit, _get_parameter_set


class LinCombEstimatorGradient(BaseEstimatorGradient):
    """Compute the gradients of the expectation values.
    This method employs a linear combination of unitaries [1].

    **Reference:**
    [1] Schuld et al., Evaluating analytic gradients on quantum hardware, 2018
    `arXiv:1811.11184 <https://arxiv.org/pdf/1811.11184.pdf>`_
    """

    SUPPORTED_GATES = [
        "rx",
        "ry",
        "rz",
        "rzx",
        "rzz",
        "ryy",
        "rxx",
        "cx",
        "cy",
        "cz",
        "ccx",
        "swap",
        "iswap",
        "h",
        "t",
        "s",
        "sdg",
        "x",
        "y",
        "z",
    ]

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
        self._lin_comb_cache = {}
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
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        jobs, metadata_ = [], []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # a set of parameters to be differentiated
            parameter_set = _get_parameter_set(circuit, parameters_)
            meta = {"parameters": [p for p in circuit.parameters if p in parameter_set]}
            meta["derivative_type"] = self._derivative_type
            metadata_.append(meta)
            # prepare circuits for the gradient of the specified parameters
            circuit_key = _circuit_key(circuit)
            if circuit_key not in self._lin_comb_cache:
                self._lin_comb_cache[circuit_key] = _make_lin_comb_gradient_circuit(circuit)
            lin_comb_circuits = self._lin_comb_cache[circuit_key]
            gradient_circuits = []
            for param in circuit.parameters:
                if param not in parameter_set:
                    continue
                gradient_circuits.append(lin_comb_circuits[param])
            n = len(gradient_circuits)
            # Make the observable as :class:`~qiskit.quantum_info.SparsePauliOp` and
            # add an ancillary operator to compute the gradient.
            observable = init_observable(observable)
            observable_1, observable_2 = _make_lin_comb_observables(
                observable, self._derivative_type
            )
            # if its derivative type is `DerivativeType.COMPLEX`, calculate the gradient
            # of the real and imaginary parts separately.
            if self._derivative_type == DerivativeType.COMPLEX:
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

        try:
            results = [job.result() for job in jobs]
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job failed.") from exc
        # compute the gradients
        gradients = []
        for i, result in enumerate(results):
            gradient = np.zeros(len(metadata_[i]["parameters"]), dtype="complex")
            if metadata_[i]["derivative_type"] == DerivativeType.COMPLEX:
                n = len(result.values) // 2  # is always a multiple of 2
                gradient = result.values[:n] + 1j * result.values[n:]
            else:
                gradient = np.real(result.values)
            gradients.append(gradient)
        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata_, options=opt)


def _make_lin_comb_observables(
    observable: BaseOperator | PauliSumOp,
    derivative_type: DerivativeType,
) -> tuple[BaseOperator | PauliSumOp, BaseOperator | PauliSumOp | None]:
    """Make the observable with an ancillary operator for the linear combination gradient.

    Args:
        observable: The observable.
        derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
            ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``.

    Returns:
        The observable with an ancillary operator for the linear combination gradient.

    Raises:
        ValueError: If the derivative type is not supported.
    """
    if derivative_type == DerivativeType.REAL:
        return observable.expand(SparsePauliOp.from_list([("Z", 1)])), None
    elif derivative_type == DerivativeType.IMAG:
        return observable.expand(SparsePauliOp.from_list([("Y", -1)])), None
    elif derivative_type == DerivativeType.COMPLEX:
        return observable.expand(SparsePauliOp.from_list([("Z", 1)])), observable.expand(
            SparsePauliOp.from_list([("Y", -1)])
        )
    else:
        raise ValueError(f"Derivative type {derivative_type} is not supported.")
