# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

import warnings
from abc import ABC, abstractmethod
from qiskit.aqua.utils import get_entangler_map, validate_entangler_map


class FeatureMap(ABC):
    """Base class for FeatureMap.

    This method should initialize the module and
    use an exception if a component of the module is not
    available.
    """

    @abstractmethod
    def __init__(self) -> None:
        warnings.warn('The FeatureMap class is deprecated as of Qiskit Aqua 0.9.0 and will be '
                      'removed no earlier than 3 months after the release date. You should use '
                      'plain QuantumCircuits instead, or data preparation circuits from '
                      'qiskit.circuit.library or qiskit.ml.circuit.library.',
                      DeprecationWarning, stacklevel=2)
        self._num_qubits = 0
        self._feature_dimension = 0
        self._support_parameterized_circuit = False

    @abstractmethod
    def construct_circuit(self, x, qr=None, inverse=False):
        """Construct the variational form, given its parameters.

        Args:
            x (numpy.ndarray[float]): 1-D array, data
            qr (QuantumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): whether or not inverse the circuit

        Returns:
            QuantumCircuit: a quantum circuit.
        """
        raise NotImplementedError()

    @staticmethod
    def get_entangler_map(map_type, num_qubits):
        """ get entangle map """
        return get_entangler_map(map_type, num_qubits)

    @staticmethod
    def validate_entangler_map(entangler_map, num_qubits):
        """ validate entangler map """
        return validate_entangler_map(entangler_map, num_qubits)

    @property
    def feature_dimension(self):
        """ returns feature dimension """
        return self._feature_dimension

    @property
    def num_qubits(self):
        """ returns number of qubits """
        return self._num_qubits

    @property
    def support_parameterized_circuit(self):
        """ returns whether or not the sub-class support parameterized circuit """
        return self._support_parameterized_circuit

    @support_parameterized_circuit.setter
    def support_parameterized_circuit(self, new_value):
        """ set whether or not the sub-class support parameterized circuit """
        self._support_parameterized_circuit = new_value
