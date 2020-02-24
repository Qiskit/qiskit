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

"""Global X phases and parameterized problem hamiltonian."""

from typing import Optional
from functools import reduce

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator, op_converter
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states import InitialState

# pylint: disable=invalid-name


class QAOAVarForm(VariationalForm):
    """Global X phases and parameterized problem hamiltonian."""

    def __init__(self, cost_operator: WeightedPauliOperator,
                 p: int,
                 initial_state: Optional[InitialState] = None,
                 mixer_operator: Optional[WeightedPauliOperator] = None):
        """
        Constructor, following the QAOA paper https://arxiv.org/abs/1411.4028

        Args:
            cost_operator: The operator representing the cost of
                            the optimization problem,
                            denoted as U(B, gamma) in the original paper.
            p: The integer parameter p, which determines the depth of the circuit,
                as specified in the original paper.
            initial_state: An optional initial state to use.
            mixer_operator: An optional custom mixer operator to use instead of
                            the global X-rotations,
                            denoted as U(B, beta) in the original paper.
        Raises:
            TypeError: invalid input
        """
        super().__init__()
        cost_operator = op_converter.to_weighted_pauli_operator(cost_operator)
        self._cost_operator = cost_operator
        self._num_qubits = cost_operator.num_qubits
        self._p = p
        self._initial_state = initial_state
        self._num_parameters = 2 * p
        self._bounds = [(0, np.pi)] * p + [(0, 2 * np.pi)] * p
        self._preferred_init_points = [0] * p * 2

        # prepare the mixer operator
        v = np.zeros(self._cost_operator.num_qubits)
        ws = np.eye(self._cost_operator.num_qubits)
        if mixer_operator is None:
            self._mixer_operator = reduce(
                lambda x, y: x + y,
                [
                    WeightedPauliOperator([[1.0, Pauli(v, ws[i, :])]])
                    for i in range(self._cost_operator.num_qubits)
                ]
            )
        else:
            if not isinstance(mixer_operator, WeightedPauliOperator):
                raise TypeError('The mixer should be a qiskit.aqua.operators.WeightedPauliOperator '
                                + 'object, found {} instead'.format(type(mixer_operator)))
            self._mixer_operator = mixer_operator
        self.support_parameterized_circuit = True

    def construct_circuit(self, parameters, q=None):
        """ construct circuit """
        angles = parameters
        if not len(angles) == self.num_parameters:
            raise ValueError('Incorrect number of angles: expecting {}, but {} given.'.format(
                self.num_parameters, len(angles)
            ))

        # initialize circuit, possibly based on given register/initial state
        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        circuit.u2(0, np.pi, q)
        for idx in range(self._p):
            beta, gamma = angles[idx], angles[idx + self._p]
            circuit += self._cost_operator.evolve(
                evo_time=gamma, num_time_slices=1, quantum_registers=q
            )
            circuit += self._mixer_operator.evolve(
                evo_time=beta, num_time_slices=1, quantum_registers=q
            )
        return circuit

    @property
    def setting(self):
        """ returns setting """
        ret = "Variational Form: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret
