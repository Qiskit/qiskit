# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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

from collections.abc import Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import init_observable, _circuit_key
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..base.base_estimator_gradient import BaseEstimatorGradient
from ..base.estimator_gradient_result import EstimatorGradientResult
from ..utils import DerivativeType, _make_lin_comb_gradient_circuit, _make_lin_comb_observables

from ...exceptions import AlgorithmError


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
        self._lin_comb_cache: dict[tuple, dict[Parameter, QuantumCircuit]] = {}
        super().__init__(estimator, options, derivative_type=derivative_type)

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
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        job_circuits, job_observables, job_param_values, metadata = [], [], [], []
        all_n = []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # Prepare circuits for the gradient of the specified parameters.
            meta = {"parameters": parameters_}
            circuit_key = _circuit_key(circuit)
            if circuit_key not in self._lin_comb_cache:
                # Cache the circuits for the linear combination of unitaries.
                # We only cache the circuits for the specified parameters in the future.
                self._lin_comb_cache[circuit_key] = _make_lin_comb_gradient_circuit(
                    circuit, add_measurement=False
                )
            lin_comb_circuits = self._lin_comb_cache[circuit_key]
            gradient_circuits = []
            for param in parameters_:
                gradient_circuits.append(lin_comb_circuits[param])
            n = len(gradient_circuits)
            # Make the observable as :class:`~qiskit.quantum_info.SparsePauliOp` and
            # add an ancillary operator to compute the gradient.
            observable = init_observable(observable)
            observable_1, observable_2 = _make_lin_comb_observables(
                observable, self._derivative_type
            )
            # If its derivative type is `DerivativeType.COMPLEX`, calculate the gradient
            # of the real and imaginary parts separately.
            meta["derivative_type"] = self.derivative_type
            metadata.append(meta)
            # Combine inputs into a single job to reduce overhead.
            if self._derivative_type == DerivativeType.COMPLEX:
                job_circuits.extend(gradient_circuits * 2)
                job_observables.extend([observable_1] * n + [observable_2] * n)
                job_param_values.extend([parameter_values_] * 2 * n)
                all_n.append(2 * n)
            else:
                job_circuits.extend(gradient_circuits)
                job_observables.extend([observable_1] * n)
                job_param_values.extend([parameter_values_] * n)
                all_n.append(n)

        # Run the single job with all circuits.
        job = self._estimator.run(
            job_circuits,
            job_observables,
            job_param_values,
            **options,
        )
        try:
            results = job.result()
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        for n in all_n:
            # this disable is needed as Pylint does not understand derivative_type is a property if
            # it is only defined in the base class and the getter is in the child
            # pylint: disable=comparison-with-callable
            if self.derivative_type == DerivativeType.COMPLEX:
                gradient = np.zeros(n // 2, dtype="complex")
                gradient.real = results.values[partial_sum_n : partial_sum_n + n // 2]
                gradient.imag = results.values[partial_sum_n + n // 2 : partial_sum_n + n]

            else:
                gradient = np.real(results.values[partial_sum_n : partial_sum_n + n])
            partial_sum_n += n
            gradients.append(gradient)

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata, options=opt)
