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
Backend wrapper classes
"""

from __future__ import annotations

import logging
from typing import Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import Counts, Result

logger = logging.getLogger(__name__)


class BackendWrapper:
    """
    Backend wrapper
    """

    def __init__(self, backend: Backend):
        """
        Args:
            backend: backend
        """
        self._backend = backend

    @property
    def backend(self) -> Backend:
        """
        Returns:
            backend
        """
        return self._backend

    def run(self, circuits: Union[QuantumCircuit, list[QuantumCircuit]], **options) -> Result:
        """Executes a quantum circuit or a set of quantum circuits and returns results

        Args:
            circuits: quantum circuits to be executed
            options: options for `backend.run`
        Returns:
            Result: the result of a job
        """
        job = self._backend.run(circuits, **options)
        return job.result()

    @classmethod
    def from_backend(cls, backend: Union[Backend, BackendWrapper]) -> BackendWrapper:
        """Generate `BackendWrapper` instance from a backend

        Args:
            backend: a backend or a backend wrapper

        Returns:
            a backend wrapper object
        """
        if isinstance(backend, Backend):
            return cls(backend)
        return backend

    def get_counts(self, results: list[Result]) -> list[list[Counts]]:
        """Returns a list of lists of counts from a list of results.

        Args:
            results: a list of results

        Returns:
            a list of lists of counts
        """
        ret = []
        for result in results:
            counts = result.get_counts()
            if isinstance(counts, list):
                ret.append(counts)
            else:
                # if there is only a circuit executed, the result is a `Counts` not `list[Counts]`.
                ret.append([counts])
        return ret
