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
from .containers import EstimatorPubLike, PrimitiveResult, PubResult
from .containers.estimator_pub import EstimatorPub
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction


class StatevectorEstimator(BaseEstimatorV2):
    """
    Simple implementation of :class:`BaseEstimatorV2` with full state vector simulation.

    This class is implemented via :class:`~.Statevector` which turns provided circuits into
    pure state vectors. These states are subsequently acted on by :class:~.SparsePauliOp`,
    which implies that, at present, this implementation is only compatible with Pauli-based
    observables.
    """

    def __init__(
        self, *, default_precision: float = 0.0, seed: np.random.Generator | int | None = None
    ):
        """
        Args:
            default_precision: The default precision for the estimator if not specified during run.
            seed: The seed or Generator object for random number generation.
                If None, a random seeded default RNG will be used.
        """
        self._default_precision = default_precision
        self._seed = seed

    @property
    def default_precision(self) -> int:
        """Return the default shots"""
        return self._default_precision

    @property
    def seed(self) -> np.random.Generator | int | None:
        """Return the seed or Generator object for random number generation."""
        return self._seed

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs])

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        rng = np.random.default_rng(self._seed)
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision
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
            if precision != 0:
                if not np.isreal(expectation_value):
                    raise ValueError("Given operator is not Hermitian and noise cannot be added.")
                expectation_value = rng.normal(expectation_value, precision)
            evs[index] = expectation_value
        data_bin_cls = self._make_data_bin(pub)
        data_bin = data_bin_cls(evs=evs, stds=stds)
        return PubResult(data_bin, metadata={"precision": precision})
