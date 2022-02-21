# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Expectation value class
"""

# pylint: disable=no-name-in-module, import-error

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Result
from qiskit.transpiler import PassManager
from qiskit.utils import optionals as _optionals

from ..framework.utils import PauliSumOp
from ..results import EstimatorResult
from .base_estimator import BaseEstimator

if _optionals.HAS_AER:
    from qiskit.providers.aer.library import SaveExpectationValueVariance


class ExactEstimator(BaseEstimator):
    """
    Calculates the expectation value exactly (i.e. without sampling error).
    """

    def __init__(
        self,
        circuits: Union[QuantumCircuit, list[Union[QuantumCircuit]]],
        observables: Union[BaseOperator, PauliSumOp, list[Union[BaseOperator, PauliSumOp]]],
        backend: Backend,
        transpile_options: Optional[dict] = None,
        bound_pass_manager: Optional[PassManager] = None,
    ):
        if not _optionals.HAS_AER:
            raise MissingOptionalLibraryError(
                libname="qiskit-aer",
                name="Aer provider",
                pip_install="pip install qiskit-aer",
            )

        super().__init__(
            circuits=circuits,
            observables=observables,
            backend=backend,
            transpile_options=transpile_options,
            bound_pass_manager=bound_pass_manager,
        )

    def _preprocessing(
        self, circuits: list[QuantumCircuit], observables: list[SparsePauliOp]
    ) -> list[QuantumCircuit]:
        preprocessed_circuits = []
        for group in self._grouping:
            circuit_copy = circuits[group.circuit_index].copy()
            circuit_copy.append(
                SaveExpectationValueVariance(operator=observables[group.observable_index]),
                qargs=range(circuit_copy.num_qubits),
            )
            preprocessed_circuits.append(circuit_copy)
        return preprocessed_circuits

    def _postprocessing(self, result: Result) -> EstimatorResult:

        # TODO: validate

        expvals = []
        variances = []
        if isinstance(result, Result):
            for r in result.results:
                expval, variance = r.data.expectation_value_variance
                expvals.append(expval)
                variances.append(variance)
        else:
            raise TypeError("TODO")

        return EstimatorResult(
            np.array(expvals, dtype=np.float64), np.array(variances, dtype=np.float64)
        )
