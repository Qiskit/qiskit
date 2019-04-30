# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
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

from abc import abstractmethod

from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.utils import get_entangler_map, validate_entangler_map


class VariationalForm(Pluggable):

    """Base class for VariationalForms.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self._num_parameters = 0
        self._num_qubits = 0
        self._bounds = list()
        pass

    @classmethod
    def init_params(cls, params):
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        args = {k: v for k, v in var_form_params.items() if k != 'name'}

        # We pass on num_qubits to initial state since we know our dependent needs this
        init_state_params = params.get(Pluggable.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = var_form_params['num_qubits']
        args['initial_state'] = get_pluggable_class(PluggableType.INITIAL_STATE,
                                                    init_state_params['name']).init_params(params)

        return cls(**args)

    @abstractmethod
    def construct_circuit(self, parameters, q=None):
        """Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray[float]): circuit parameters.
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            A quantum circuit.
        """
        raise NotImplementedError()

    @property
    def num_parameters(self):
        """Number of parameters of the variational form.

        Returns:
            An integer indicating the number of parameters.
        """
        return self._num_parameters

    @property
    def num_qubits(self):
        """Number of qubits of the variational form.

        Returns:
            An integer indicating the number of qubits.
        """
        return self._num_qubits

    @property
    def parameter_bounds(self):
        """Parameter bounds.

        Returns:
            A list of pairs indicating the bounds, as (lower,
            upper). None indicates an unbounded parameter in the
            corresponding direction. If None is returned, problem is
            fully unbounded.
        """
        return self._bounds

    @property
    def setting(self):
        ret = "Variational Form: {}\n".format(self._configuration['name'])
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    @property
    def preferred_init_points(self):
        return None

    @staticmethod
    def get_entangler_map(map_type, num_qubits):
        return get_entangler_map(map_type, num_qubits)

    @staticmethod
    def validate_entangler_map(entangler_map, num_qubits):
        return validate_entangler_map(entangler_map, num_qubits)
