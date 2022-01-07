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
Readout error mitigation backend wrapper with periodical calibration
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Union

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import BaseReadoutMitigator, Counts, Result

from .backend_wrapper import BackendWrapper

logger = logging.getLogger(__name__)


class ReadoutErrorMitigation(BackendWrapper):
    """
    Readout error mitigation backend wrapper with periodical calibration
    """

    def __init__(
        self,
        backend: Backend,
        method: str,
        refresh: float,
        qubits: list[int],
        **cal_options,
    ):
        """
        Args:
            backend: backend
            method: error mitigation name ('local', 'correlated' or 'mthree')
            refresh: calibrate the error mitigation matrix every `refresh` seconds
            qubits: a list of qubits to apply error mitigation
            cal_options: options to be passed to the error mitigation class

        Raises:
            ValueError: if unsupported error mitigation name is given as `method`
        """
        super().__init__(backend=backend)
        if method not in ["mthree", "local", "correlated"]:
            raise ValueError(f"Unsupported mitigation method: '{method}'")
        self._method = method
        self._refresh = timedelta(seconds=refresh)
        self._time_threshold = datetime.min.replace(tzinfo=timezone.utc)
        self._qubits = qubits
        self._cal_options = cal_options

        if TYPE_CHECKING:
            from mthree import M3Mitigation
        self._mitigators: dict[datetime, Union[BaseReadoutMitigator, M3Mitigation]] = {}

    @property
    def mitigation(self):
        """
        Returns:
            error mitigation name
        """
        return self._method

    @property
    def refresh(self):
        """
        Returns:
            the refresh time of calibration
        """
        return self._refresh

    @property
    def cal_options(self):
        """
        Returns:
            options to be passed to the error mitigation class
        """
        return self._cal_options

    @staticmethod
    def _datetime(data):
        # Note: Aer's result.date is str without tzinfo, but IBMQ's result.date is datetime with tzinfo
        if isinstance(data, str):
            return datetime.fromisoformat(data).astimezone()
        return data

    def _maybe_calibrate(self):
        now = datetime.now(timezone.utc).astimezone()
        if now <= self._time_threshold:
            return
        logger.info("readout error mitigation calibration %s at %s", self._method, now)
        if self._method == "mthree":
            try:
                from mthree import M3Mitigation
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    libname="mthree",
                    name="execute",
                    pip_install="pip install mthree",
                ) from ex
            mit = M3Mitigation(self._backend)
            mit.cals_from_system(qubits=self._qubits, **self._cal_options)
            self._mitigators[now] = mit
        elif self._method in ["local", "correlated"]:
            try:
                from qiskit_experiments.library import ReadoutMitigationExperiment
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    libname="qiskit_experiments>=0.3.0",
                    name="execute",
                    pip_install="pip install qiskit-experiments",
                ) from ex

            exp = ReadoutMitigationExperiment(qubits=self._qubits, method=self._method)
            result = exp.run(self._backend, **self._cal_options).block_for_results()
            mit = result.analysis_results(0).value
            self._mitigators[now] = mit
        self._time_threshold = now + self._refresh

    def _apply_mitigation(self, result: Result) -> list[Counts]:
        result_dt = self._datetime(result.date)
        mitigators = [
            (abs(date - result_dt), date, mitigator) for date, mitigator in self._mitigators.items()
        ]
        _, min_date, mitigator = min(mitigators, key=lambda e: e[0])
        logger.info("apply mitigation data at %s", min_date)
        counts = result.get_counts()
        if isinstance(counts, Counts):
            counts = [counts]
        shots = [count.shots() for count in counts]
        if self._method == "mthree":
            quasis = mitigator.apply_correction(counts, self._qubits)  # type: ignore
        else:
            quasis = [mitigator.quasi_probabilities(count, self._qubits) for count in counts]
        ret = []
        for quasi, shot in zip(quasis, shots):
            ret.append(Counts({key: val * shot for key, val in quasi.items()}))
        return ret

    def get_counts(self, results: list[Result]) -> list[list[Counts]]:
        return [self._apply_mitigation(result) for result in results]

    def run(self, circuits: Union[QuantumCircuit, list[QuantumCircuit]], **options) -> Result:
        self._maybe_calibrate()
        result = self._backend.run(circuits, **options).result()
        self._maybe_calibrate()
        return result
