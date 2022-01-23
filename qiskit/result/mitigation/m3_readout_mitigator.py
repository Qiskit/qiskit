# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Readout mitigator class based on the 1-qubit local tensored mitigation method
"""


from typing import Optional, List, Tuple, Iterable, Callable, Union
import numpy as np

from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .mthree import M3Mitigation
from .local_readout_mitigator import LocalReadoutMitigator


class M3ReadoutMitigator(LocalReadoutMitigator):
    """1-qubit tensor product readout error mitigator based on the
    matrix-free measurement mitigator (M3).

    Mitigates :meth:`expectation_value` and :meth:`quasi_probabilities`.
    The mitigator should either be calibrated using qiskit experiments,
    or calculated directly from the backend properties.
    This mitigation method should be used in case the readout errors of the qubits
    are assumed to be uncorrelated. For *N* qubits there are *N* mitigation matrices,
    each of size :math:`2 x 2` and the mitigation complexity is :math:`O(2^N)`,
    so it is more efficient than the :class:`CorrelatedReadoutMitigator` class.
    """

    def __init__(
        self,
        amats: Optional[List[np.ndarray]] = None,
        qubits: Optional[Iterable[int]] = None,
        backend=None,
    ):
        """Initialize a M3ReadoutMitigator

        Args:
            amats: Optional, list of single-qubit readout error assignment matrices.
            qubits: Optional, the measured physical qubits for mitigation.
            backend: Optional, backend name.

        Raises:
            QiskitError: matrices sizes do not agree with number of qubits
        """
        if amats is None:
            amats = self._from_backend(backend, qubits)
        else:
            amats = [np.asarray(amat, dtype=float) for amat in amats]
        for amat in amats:
            if np.any(amat < 0) or not np.allclose(np.sum(amat, axis=0), 1):
                raise QiskitError(
                    "Assignment matrix columns must be valid probability distributions"
                )


        if qubits is None:
            self._num_qubits = len(amats)
            self._qubits = range(self._num_qubits)
            qubits = self._qubits
        else:
            if len(qubits) != len(amats):
                raise QiskitError(
                    "The number of given qubits ({}) is different than the number of qubits "
                    "inferred from the matrices ({})".format(len(qubits), len(amats))
                )
            self._qubits = qubits
            self._num_qubits = len(self._qubits)

        max_qubit_index = max(qubits)
        m3_amats = [None] * (max_qubit_index + 1)
        for amat, qubit in zip(amats, qubits):
            m3_amats[qubit] = amat

        self._m3_mitigator = M3Mitigation(m3_amats)
        self._assignment_mats = amats
        self._qubit_index = dict(zip(self._qubits, range(self._num_qubits)))
        self.compute_gammas()

    def quasi_probabilities(
        self,
        data: Counts,
        qubits: Iterable[int] = None,
        clbits: Optional[List[int]] = None,
        shots: Optional[int] = None,
    ) -> QuasiDistribution:
        if qubits is None:
            qubits = self._qubits
        quasi_probs = self._m3_mitigator._apply_correction(data, qubits)
        counts = dict(data)
        shots = sum(counts.values())
        # TODO: verify this estimate can be used here
        quasi_probs._stddev_upper_bound = self.stddev_upper_bound(shots, qubits)
        return quasi_probs

    def expectation_value(
            self,
            data: Counts,
            diagonal: Union[Callable, dict, str, np.ndarray] = None,
            qubits: Iterable[int] = None,
            clbits: Optional[List[int]] = None,
            shots: Optional[int] = None,
    ) -> Tuple[float, float]:
        pass # placeholder