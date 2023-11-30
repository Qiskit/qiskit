# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic.types import PositiveInt

from qiskit.quantum_info import SparsePauliOp, Statevector

from .base import BaseEstimatorV2
from .containers import (
    BasePrimitiveOptions,
    BasePrimitiveOptionsLike,
    EstimatorPubs,
    EstimatorPubsLike,
    PrimitiveResult,
    PubsResult,
    make_data_bin,
)
from .containers.dataclasses import mutable_dataclass
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction


@mutable_dataclass
class ExecutionOptions(BasePrimitiveOptions):
    """Options for execution."""

    shots: Optional[PositiveInt] = None
    seed: Optional[Union[int, np.random.Generator]] = None


@mutable_dataclass
class Options(BasePrimitiveOptions):
    """Options for the primitives.

    Args:
        execution: Execution time options. See :class:`ExecutionOptions` for all available options.
    """

    execution: ExecutionOptions = Field(default_factory=ExecutionOptions)


class Estimator(BaseEstimatorV2):
    """
    Simple implementation of :class:`BaseEstimatorV2` with Statevector.

    :Execution Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the exact expectation
          values. Otherwise, it samples from normal distributions with standard errors as standard
          deviations using normal distribution approximation.

        - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the normal distribution. If shots is None,
          this option is ignored.
    """

    _options_class = Options
    options: Options

    def __init__(self, *, options: Optional[BasePrimitiveOptionsLike] = None):
        """
        Args:
            options: Options including shots, seed.
        """
        if options is None:
            options = Options()
        elif not isinstance(options, BasePrimitiveOptions):
            options = Options(**options)
        super().__init__(options=options)

    def run(self, pubs: Iterable[EstimatorPubsLike]) -> PrimitiveJob[PrimitiveResult[PubsResult]]:
        coerced_pubs = [EstimatorPubs.coerce(pub) for pub in pubs]

        for pub in coerced_pubs:
            pub.validate()

        job: PrimitiveJob[PrimitiveResult[PubsResult]] = PrimitiveJob(self._run_pubs, coerced_pubs)
        job.submit()
        return job

    def _run_pubs(self, pubs: list[EstimatorPubs]) -> PrimitiveResult[PubsResult]:
        shots = self.options.execution.shots

        rng = _get_rng(self.options.execution.seed)

        results = []
        for pub in pubs:
            circuit = pub.circuit
            observables = pub.observables
            parameter_values = pub.parameter_values
            bound_circuits = parameter_values.bind_all(circuit)

            bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)
            evs = np.zeros_like(bc_circuits, dtype=np.complex128)
            stds = np.zeros_like(bc_circuits, dtype=np.complex128)
            for index in np.ndindex(*bc_circuits.shape):
                bound_circuit = bc_circuits[index]
                observable = bc_obs[index]

                final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
                paulis, coeffs = zip(*observable.items())
                obs = SparsePauliOp(paulis, coeffs)
                # TODO: support non Pauli operators
                expectation_value = np.real_if_close(final_state.expectation_value(obs))
                if shots is None:
                    standard_error = 0
                else:
                    sq_obs = (obs @ obs).simplify(atol=0)
                    sq_exp_val = np.real_if_close(final_state.expectation_value(sq_obs))
                    variance = sq_exp_val - expectation_value**2
                    variance = max(variance, 0)
                    standard_error = np.sqrt(variance / shots)
                    expectation_value = rng.normal(expectation_value, standard_error)
                evs[index] = expectation_value
                stds[index] = standard_error
            data_bin_cls = make_data_bin(
                [("evs", NDArray[np.complex128]), ("stds", NDArray[np.complex128])],
                shape=bc_circuits.shape,
            )
            data_bin = data_bin_cls(evs=evs, stds=stds)
            results.append(PubsResult(data_bin, metadata={"shots": shots}))
        return PrimitiveResult(results)


def _get_rng(seed):
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    return rng
