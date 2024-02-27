# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Estimator Converter class
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from qiskit.quantum_info import SparsePauliOp

from .base import BaseEstimatorV1, BaseEstimatorV2
from .containers import EstimatorPubLike, PrimitiveResult, PubResult
from .containers.estimator_pub import EstimatorPub
from .primitive_job import PrimitiveJob


class EstimatorV2Converter(BaseEstimatorV2):
    """A converter class that takes a :class:`~.BaseEstimatorV1` instance and wraps it in a
    :class:`~.BaseEstimatorV2` interface.
    """

    def __init__(
        self,
        estimator: BaseEstimatorV1,
    ):
        """Initialize a EstimatorV2Converter converter instance based on a BaseEstiamtorV1 instance.

        Args:
            estimator: The input :class:`~.BaseEstimatorV1` based backend to wrap in a
                :class:`~.BaseEstimatorV2` interface.
        """
        self._estimatorv1 = estimator

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> BasePrimitiveJob[PrimitiveResult[PubResult]]:
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs])

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        out_shape = np.broadcast_shapes(pub.observables.shape, pub.parameter_values.shape)
        param_broad = np.broadcast_to(pub.parameter_values, out_shape)
        param_object = np.zeros(pub.parameter_values.shape, dtype=object)
        param_array = pub.parameter_values.as_array(circuit.parameters)
        for idx in np.ndindex(pub.parameter_values.shape):
            param_object[idx] = param_array[idx].tolist()

        obs_list = []
        param_list = []
        for obs, param in np.broadcast(pub.observables, param_object):
            spo = SparsePauliOp.from_list(list(obs.items()))
            obs_list.append(spo)
            param_list.append(param)

        size = len(obs_list)
        result = self._estimatorv1.run([circuit] * size, obs_list, param_list).result()
        evs = result.values.reshape(out_shape)
        stds_list = [
            np.sqrt(dat.get("variance", 0) / dat.get("shots", 1)) for dat in result.metadata
        ]
        stds = np.array(stds_list).reshape(out_shape)
        data_bin = self._make_data_bin(pub)(evs=evs, stds=stds)
        return PubResult(data_bin, metadata={"precision": precision})
