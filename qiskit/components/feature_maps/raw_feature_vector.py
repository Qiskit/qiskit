# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Raw Feature Vector feature map.
"""

import logging
import warnings

import numpy as np
from qiskit import QuantumCircuit  # pylint: disable=unused-import

from qiskit.aqua.utils.arithmetic import next_power_of_2_base
from qiskit.aqua.circuits import StateVectorCircuit
from qiskit.aqua.utils.validation import validate_min
from .feature_map import FeatureMap

logger = logging.getLogger(__name__)


class RawFeatureVector(FeatureMap):
    """
    Raw Feature Vector feature map.

    The Raw Feature Vector can be directly used as a feature map, where the raw feature vectors
    will be automatically padded with ending 0s as necessary, to make sure vector length
    is a power of 2, and normalized such that it can be treated and used
    as an initial quantum state vector.
    """

    def __init__(self, feature_dimension: int = 2) -> None:
        """
        Args:
            feature_dimension: The feature dimension, has a minimum value of 1.
        """
        validate_min('feature_dimension', feature_dimension, 1)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            super().__init__()

        warnings.warn('The RawFeatureVector class has moved to qiskit.ml.circuit.library and '
                      'subclasses the QuantumCircuit. This class, in qiskit.aqua.components, is '
                      'deprecated as of Qiskit Aqua 0.9.0 and will be removed no earlier than 3 '
                      'months after the release date.',
                      DeprecationWarning, stacklevel=2)

        self._feature_dimension = feature_dimension
        self._num_qubits = next_power_of_2_base(feature_dimension)

    def construct_circuit(self, x, qr=None, inverse=False):
        """
        Construct the second order expansion based on given data.

        Args:
            x (numpy.ndarray): 1-D to-be-encoded data.
            qr (QuantumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): inverse
        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        Raises:
            TypeError: invalid input
            ValueError: invalid input
        """
        if len(x) != self._feature_dimension:
            raise ValueError("Unexpected feature vector dimension.")

        state_vector = np.pad(x, (0, (1 << self.num_qubits) - len(x)), 'constant')

        svc = StateVectorCircuit(state_vector)
        return svc.construct_circuit(register=qr)
