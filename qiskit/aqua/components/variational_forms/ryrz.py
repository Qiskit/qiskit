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

"""Layers of Y+Z rotations followed by entangling gates."""

import warnings
from typing import Optional, List
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.components.initial_states import InitialState
from .variational_form import VariationalForm


class RYRZ(VariationalForm):
    r"""DEPRECATED. The RYRZ Variational Form.

    The RYRZ trial wave function is layers of :math:`y` plus :math:`z` rotations with entanglements.
    When none of qubits are unentangled to other qubits, the number of optimizer parameters this
    form creates and uses is given by :math:`q \times (d + 1) \times 2`, where :math:`q` is the
    total number of qubits and :math:`d` is the depth of the circuit.
    Nonetheless, in some cases, if an `entangler_map` does not include all qubits, that is, some
    qubits are not entangled by other qubits. The number of parameters is reduced by
    :math:`d \times q' \times 2` where :math:`q'` is the number of unentangled qubits.
    This is because adding more parameters to the unentangled qubits only introduce overhead
    without bringing any benefit; furthermore, theoretically, applying multiple Ry and Rz gates
    in a row can be reduced to a single Ry gate and one Rz gate with the summed rotation angles.

    See :class:`RY` for more detail on `entangler_map` and `entanglement` which apply here too
    but note RYRZ only supports 'full' and 'linear' values.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 3,
                 entangler_map: Optional[List[List[int]]] = None,
                 entanglement: str = 'full',
                 initial_state: Optional[InitialState] = None,
                 entanglement_gate: str = 'cz',
                 skip_unentangled_qubits: bool = False) -> None:
        """
        Args:
            num_qubits: Number of qubits, has a minimum value of 1.
            depth: Number of rotation layers, has a minimum value of 1.
            entangler_map: Describe the connectivity of qubits, each list pair describes
                [source, target], or None for as defined by `entanglement`.
                Note that the order is the list is the order of applying the two-qubit gate.
            entanglement: ('full' | 'linear') overridden by 'entangler_map` if its
                provided. 'full' is all-to-all entanglement, 'linear' is nearest-neighbor.
            initial_state: An initial state object
            entanglement_gate: ('cz' | 'cx')
            skip_unentangled_qubits: Skip the qubits not in the entangler_map
        """
        warnings.warn('The qiskit.aqua.components.variational_forms.RYRZ object is deprecated as '
                      'of 0.7.0 and will be removed no sooner than 3 months after the release. You '
                      'should use qiskit.circuit.library.EfficientSU2 (uses CX entangling) or '
                      'qiskit.circuit.library.TwoLocal instead.',
                      DeprecationWarning, stacklevel=2)

        validate_min('num_qubits', num_qubits, 1)
        validate_min('depth', depth, 1)
        validate_in_set('entanglement', entanglement, {'full', 'linear'})
        validate_in_set('entanglement_gate', entanglement_gate, {'cz', 'cx'})
        super().__init__()
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        # determine the entangled qubits
        all_qubits = []
        for src, targ in self._entangler_map:
            all_qubits.extend([src, targ])
        self._entangled_qubits = sorted(list(set(all_qubits)))
        self._initial_state = initial_state
        self._entanglement_gate = entanglement_gate
        self._skip_unentangled_qubits = skip_unentangled_qubits

        # for the first layer
        self._num_parameters = len(self._entangled_qubits) * 2 if self._skip_unentangled_qubits \
            else self._num_qubits * 2
        # for repeated block
        self._num_parameters += len(self._entangled_qubits) * depth * 2
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._support_parameterized_circuit = True

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (Union(numpy.ndarray, list[Parameter], ParameterVector)): circuit parameters
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        param_idx = 0
        for qubit in range(self._num_qubits):
            if not self._skip_unentangled_qubits or qubit in self._entangled_qubits:
                circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                circuit.u1(parameters[param_idx + 1], q[qubit])  # rz
                param_idx += 2

        for _ in range(self._depth):
            circuit.barrier(q)
            for src, targ in self._entangler_map:
                if self._entanglement_gate == 'cz':
                    circuit.u2(0.0, np.pi, q[targ])  # h
                    circuit.cx(q[src], q[targ])
                    circuit.u2(0.0, np.pi, q[targ])  # h
                else:
                    circuit.cx(q[src], q[targ])
            circuit.barrier(q)
            for qubit in self._entangled_qubits:
                circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                circuit.u1(parameters[param_idx + 1], q[qubit])  # rz
                param_idx += 2
        circuit.barrier(q)

        return circuit
