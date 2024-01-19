# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Estimator class
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from qiskit.quantum_info import SparsePauliOp, Statevector

from .base import BaseEstimatorV2
from .containers import EstimatorPub, EstimatorPubLike, PrimitiveResult, PubResult
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction


class StatevectorEstimator(BaseEstimatorV2):
    """
    Simple implementation of :class:`BaseEstimatorV2` with Statevector.
    """

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        job: PrimitiveJob[PrimitiveResult[PubResult]] = PrimitiveJob(self._run, pubs, precision)
        job._submit()
        return job

    def _run(
        self, pubs: Iterable[EstimatorPub], precision: float | None
    ) -> PrimitiveResult[PubResult]:
        if precision is not None and precision > 0:
            raise ValueError("precision must be None or 0 for StatevectorEstimator.")

        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        results = []
        for pub in coerced_pubs:
            if pub.precision is not None and pub.precision > 0:
                raise ValueError("precision in pub must be None or 0 for StatevectorEstimator.")
            circuit = pub.circuit
            observables = pub.observables
            parameter_values = pub.parameter_values
            bound_circuits = parameter_values.bind_all(circuit)

            bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)
            evs = np.zeros_like(bc_circuits, dtype=np.float64)
            stds = np.zeros_like(bc_circuits, dtype=np.float64)
            for index in np.ndindex(*bc_circuits.shape):
                bound_circuit = bc_circuits[index]
                observable = bc_obs[index]

                final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
                paulis, coeffs = zip(*observable.items())
                obs = SparsePauliOp(paulis, coeffs)  # TODO: support non Pauli operators
                expectation_value = np.real_if_close(final_state.expectation_value(obs))
                evs[index] = expectation_value
            data_bin_cls = self._make_data_bin(pub)
            data_bin = data_bin_cls(evs=evs, stds=stds)
            results.append(PubResult(data_bin, metadata={"precision": 0}))
        return PrimitiveResult(results)
