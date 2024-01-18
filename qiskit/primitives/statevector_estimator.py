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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from qiskit.quantum_info import SparsePauliOp, Statevector

from .base import BaseEstimatorV2
from .containers import EstimatorPub, EstimatorPubLike, PrimitiveResult, PubResult
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction


@dataclass
class Options:
    """Options for estimator."""

    seed: Optional[Union[int, np.random.Generator]] = None

    def update(self, options: Options | Mapping | None = None, **kwargs):
        """Update the options."""
        if options is not None:
            if isinstance(options, Mapping):
                options_dict = options
            elif isinstance(options, Options):
                options_dict = options.__dict__
            else:
                raise TypeError(f"Type {type(options)} is not options nor Mapping class")
            for key, val in options_dict.items():
                setattr(self, key, val)

        for key, val in kwargs.items():
            setattr(self, key, val)


class Estimator(BaseEstimatorV2):
    """
    Simple implementation of :class:`BaseEstimatorV2` with Statevector.
    """

    def __init__(self, options: Options | dict | None = None):
        if options is None:
            options = Options()
        elif not isinstance(options, Options):
            options = Options(**options)
        self.options = options

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        job: PrimitiveJob[PrimitiveResult[PubResult]] = PrimitiveJob(self._run, pubs, precision)
        job._submit()
        return job

    def _run(
        self, pubs: Iterable[EstimatorPub], precision: float | None
    ) -> PrimitiveResult[PubResult]:
        precision = (
            precision or self.options.precision
        )  # TODO: switch away from options to class variable
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        rng = _get_rng(self.options.seed)

        results = []
        for pub in coerced_pubs:
            circuit = pub.circuit
            observables = pub.observables
            parameter_values = pub.parameter_values
            bound_circuits = parameter_values.bind_all(circuit)

            bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)
            evs = np.zeros_like(bc_circuits, dtype=np.float64)
            stds = np.zeros_like(bc_circuits, dtype=np.float64)
            for index in np.ndindex(*bc_circuits.shape):
                bound_circuit = parameter_values.bind(circuit, loc=index)
                observable = bc_obs[index]

                final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
                paulis, coeffs = zip(*observable.items())
                obs = SparsePauliOp(paulis, coeffs)
                # TODO: support non Pauli operators
                expectation_value = np.real_if_close(final_state.expectation_value(obs))
                if precision is None or precision == 0:
                    standard_error = 0
                else:
                    standard_error = np.sqrt(precision)
                    expectation_value = rng.normal(expectation_value, standard_error)
                evs[index] = expectation_value
                stds[index] = standard_error
            data_bin_cls = self._make_data_bin(pub)
            data_bin = data_bin_cls(evs=evs, stds=stds)
            results.append(PubResult(data_bin, metadata={"precision": precision}))
        return PrimitiveResult(results)


def _get_rng(seed):
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    return rng
