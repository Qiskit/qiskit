# -*- coding: utf-8 -*-

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
variational forms. Several types of commonly used ansatz.
"""

from typing import Optional, Union, List
# below to allow it for python 3.6.1
try:
    from typing import NoReturn
except ImportError:
    from typing import Any as NoReturn

from abc import ABC, abstractmethod
import numpy as np
from qiskit import QuantumRegister
from qiskit.aqua.utils import get_entangler_map, validate_entangler_map


class VariationalForm(ABC):

    """Base class for VariationalForms.

        This method should initialize the module and
        use an exception if a component of the module is not
        available.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self._num_parameters = 0
        self._num_qubits = 0
        self._bounds = list()  # type: List[object]
        self._preferred_init_points = None
        self._support_parameterized_circuit = False
        pass

    @abstractmethod
    def construct_circuit(self,
                          parameters: Union[List[float], np.ndarray],
                          q: Optional[QuantumRegister] = None) -> NoReturn:
        """Construct the variational form, given its parameters.

        Args:
            parameters: circuit parameters.
            q: Quantum Register for the circuit.

        Returns:
            QuantumCircuit: A quantum circuit.
        """
        raise NotImplementedError()

    @property
    def num_parameters(self):
        """Number of parameters of the variational form.

        Returns:
            int: An integer indicating the number of parameters.
        """
        return self._num_parameters

    @property
    def support_parameterized_circuit(self):
        """ Whether or not the sub-class support parameterized circuit.

        Returns:
            boolean: indicate the sub-class support parameterized circuit
        """
        return self._support_parameterized_circuit

    @support_parameterized_circuit.setter
    def support_parameterized_circuit(self, new_value):
        """ set whether or not the sub-class support parameterized circuit """
        self._support_parameterized_circuit = new_value

    @property
    def num_qubits(self):
        """Number of qubits of the variational form.

        Returns:
           int:  An integer indicating the number of qubits.
        """
        return self._num_qubits

    @property
    def parameter_bounds(self) -> List[object]:
        """Parameter bounds.

        Returns:
            list: A list of pairs indicating the bounds, as (lower,
            upper). None indicates an unbounded parameter in the
            corresponding direction. If None is returned, problem is
            fully unbounded.
        """
        return self._bounds

    @property
    def setting(self):
        """ setting """
        ret = "Variational Form: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    @property
    def preferred_init_points(self):
        """
        Return preferred init points.

        If an initial state is provided then the variational form may provide back
        this set of parameters which when used on the variational form should
        result in the overall state being that defined by the initial state
        """
        return self._preferred_init_points

    @staticmethod
    def get_entangler_map(map_type, num_qubits, offset=0):
        """ returns entangler map """
        return get_entangler_map(map_type, num_qubits, offset)

    @staticmethod
    def validate_entangler_map(entangler_map, num_qubits):
        """ validate entangler map """
        return validate_entangler_map(entangler_map, num_qubits)
