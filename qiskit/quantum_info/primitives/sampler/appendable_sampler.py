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
Appendable sampler
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import Result

from ..backends import BackendWrapper
from ..results import BaseResult, SamplerResult
from .sampler import Sampler

logger = logging.getLogger(__name__)


class AppendableSampler(Sampler):
    """Sampler class that can append results"""

    def __init__(
        self,
        backend: Union[Backend, BackendWrapper],
        circuits: Optional[Union[QuantumCircuit, list[QuantumCircuit]]] = None,
    ):
        super().__init__(backend=backend, circuits=circuits)
        self._raw_results: list[Result] = []

    # pylint: disable=arguments-differ
    def run(
        self,
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
        circuits: Optional[list[QuantumCircuit]] = None,
        append: bool = False,
        **run_options,
    ) -> SamplerResult:
        """Runs quantum circuits and returns a sampler results.

        Args:
            parameters: parameter values for parametrized quantum circuits
            circuits: quantum circuits to be sampled.
            append: if `False`, it behaves as same as `Sampler.run`. If `True`, it does not clear
                the result of the previous runs and takes the sum of the previous runs and the
                current run.
            run_options: options for `backend.run`

        Returns:
            a sampler result

        """
        if not append:
            self._raw_results.clear()
        return super().run(parameters, circuits=circuits, **run_options)

    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> SamplerResult:
        if not isinstance(result, Result):
            raise TypeError("result must be an instance of Result.")

        self._raw_results.append(result)
        quasis, shots = self._get_quasis(self._raw_results)
        metadata = [res.header.metadata for result in self._raw_results for res in result.results]

        return SamplerResult(
            quasi_dists=quasis,
            shots=shots,
            raw_results=self._raw_results,
            metadata=metadata,
        )
