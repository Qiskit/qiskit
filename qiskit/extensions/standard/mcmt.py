# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Multiple-Control, Multiple-Target Gate."""

from typing import Union, Callable, List, Tuple

import warnings
from abc import abstractproperty
from qiskit.circuit import ControlledGate, Gate, Instruction, Qubit, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.util import deprecate_arguments
from .x import XGate, CCXGate
from .y import YGate
from .z import ZGate
from .h import HGate
from .t import TGate, TdgGate
from .s import SGate, SdgGate


def _ccx_v_chain_rule(control_qubits: Union[QuantumRegister, List[Qubit]],
                      ancilla_qubits: Union[QuantumRegister, List[Qubit]],
                      reverse: bool = False) -> List[Tuple[Gate, List[Qubit], List]]:
    """Get the rule for the CCX V-chain.

    The CCX V-chain progressively computes the CCX of the control qubits and puts the final result
    in the last ancillary qubit.

    Args:
        control_qubits: The control qubits.
        ancilla_qubits: The ancilla qubits.
        reverse: If True, compute the chain down to the qubit. If False, compute upwards.

    Returns:
        The rule for the (reversed) CCX V-chain.

    Raises:
        QiskitError: If an insufficient number of ancilla qubits was provided.
    """
    if len(ancilla_qubits) == 0:
        return []

    if len(ancilla_qubits) < len(control_qubits) - 1:
        raise QiskitError('Insufficient number of ancilla qubits.')

    rule = [(CCXGate(), [control_qubits[0], control_qubits[1], ancilla_qubits[0]], [])]
    for i, j in enumerate(range(2, len(control_qubits))):
        rule += [
            (CCXGate(), [control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1]], [])
        ]

    if reverse:
        return reversed(rule)
    return rule


class MCMTGate(Gate):
    """The multi-controlled multi-target gate, for an arbitrary singly controlled target gate."""

    def __init__(self, gate: Union[Gate, Callable[[QuantumCircuit, Qubit, Qubit], Instruction]],
                 num_ctrl_qubits: int,
                 num_target_qubits: int) -> None:
        """Create a new multi-control multi-target gate.

        Args:
            gate: The gate to be applied controlled on the control qubits and applied to the target
                qubits. Can be either a Gate or a circuit method.
                If it is a callable, it will be casted to a Gate.
            num_ctrl_qubits: The number of control qubits.
            num_target_qubits: The number of target qubits.

        Raises:
            QiskitError: If the gate cannot be casted to a controlled gate.
            QiskitError: If the number of controls or targets is 0.
        """

        if num_ctrl_qubits == 0 or num_target_qubits == 0:
            raise QiskitError('Need at least one control and one target qubit.')

        self.gate = self._identify_gate(gate)
        self.num_ctrl_qubits = num_ctrl_qubits
        self.num_target_qubits = num_target_qubits

        num_qubits = num_ctrl_qubits + num_target_qubits + self.num_ancilla_qubits

        super().__init__('mcmt', num_qubits, [])

    @abstractproperty
    def num_ancilla_qubits(self):
        """Return the number of ancillas."""
        raise NotImplementedError

    def _identify_gate(self, gate):
        """Case the gate input to a gate."""
        valid_gates = {
            'ch': HGate(),
            'cx': XGate(),
            'cy': YGate(),
            'cz': ZGate(),
            'h': HGate(),
            's': SGate(),
            'sdg': SdgGate(),
            'x': XGate(),
            'y': YGate(),
            'z': ZGate(),
            't': TGate(),
            'tdg': TdgGate(),
        }
        if isinstance(gate, ControlledGate):
            base_gate = gate.base_gate
        elif isinstance(gate, Gate):
            if gate.num_qubits != 1:
                raise AttributeError('Base gate must act on one qubit only.')
            base_gate = gate
        else:
            if callable(gate):  # identify via name of the passed function
                name = gate.__name__
            elif isinstance(gate, str):
                name = gate
            else:
                raise AttributeError('Invalid gate specified: {}'.format(gate))
            base_gate = valid_gates[name]

        return base_gate

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return the controlled version of the MCMT gate."""
        if ctrl_state is None:
            return MCMTGate(self.gate,
                            self.num_ctrl_qubits + num_ctrl_qubits,
                            self.num_target_qubits)
        else:
            return super().control(num_ctrl_qubits, label, ctrl_state)


class MCMTGateNoAncilla(MCMTGate):
    """The MCMT implementation without ancilla qubits."""

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancilla qubits required."""
        return 0

    def _define(self):
        """Define the MCMT gate without ancillas."""
        if self.num_target_qubits == 1:
            # no broadcasting needed (makes for better circuit diagrams)
            broadcasted_gate = self.gate
        else:
            broadcasted = QuantumCircuit(self.num_target_qubits)
            for target in list(range(self.num_target_qubits)):
                broadcasted.append(self.gate, [target], [])
            broadcasted_gate = broadcasted.to_gate()

        mcmt_gate = broadcasted_gate.control(self.num_ctrl_qubits)
        q = QuantumRegister(self.num_qubits)
        self.definition = [(mcmt_gate, q[:], [])]

    def inverse(self):
        return MCMTGateNoAncilla(self.gate, self.num_ctrl_qubits, self.num_target_qubits)


class MCMTGateVChain(MCMTGate):
    """The MCMT implementation using the CCX V-chain."""

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancilla qubits required."""
        return max(0, self.num_ctrl_qubits - 1)

    def _define(self):
        """Define the MCMT gate."""
        definition = []
        q = QuantumRegister(self.num_qubits, 'q')

        control_qubits = q[:self.num_ctrl_qubits]
        target_qubits = q[self.num_ctrl_qubits:self.num_ctrl_qubits + self.num_target_qubits]
        ancilla_qubits = q[self.num_ctrl_qubits + self.num_target_qubits:]

        if len(ancilla_qubits) > 0:
            master_control = ancilla_qubits[-1]
        else:
            master_control = control_qubits[0]

        definition += _ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=False)
        for qubit in target_qubits:
            definition += [
                (self.gate.control(), [master_control, qubit], [])
            ]
        definition += _ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=True)

        self.definition = definition

    def inverse(self):
        return MCMTGateVChain(self.gate, self.num_ctrl_qubits, self.num_target_qubits)


# pylint:disable=unused-argument
@deprecate_arguments({'single_control_gate_fun': 'gate',
                      'q_controls': 'control_qubits',
                      'q_ancillae': 'ancilla_qubits',
                      'q_targets': 'target_qubits'})
def mcmt(self, gate, control_qubits, target_qubits, ancilla_qubits=None, mode='no-ancilla',
         *, single_control_gate_fun=None, q_controls=None, q_ancillae=None, q_targets=None):
    """Apply a multi-control, multi-target using a generic gate.

    This can also be used to implement a generic multi-Control gate, as the target could also be of
    length 1.
    """
    # for backward compatibility; the previous signature was
    # `def mcmt(self, q_controls, q_ancillae, single_control_gate_fun, q_targets, mode="basic")`
    if callable(target_qubits):
        warnings.warn('It seems you are using the old signature of the MCMT gate, which is '
                      'deprecated as of 0.13.0 and will be removed no earlier than 3 months '
                      'after the release date. The signature has been updated to '
                      'mcmct(gate, control_qubits, target_qubits, ancilla_qubits).',
                      DeprecationWarning, stacklevel=3)
        # swap arguments in the right order
        tmp = gate
        gate = target_qubits
        target_qubits = ancilla_qubits
        ancilla_qubits = control_qubits
        control_qubits = tmp

    if isinstance(control_qubits, (int, Qubit)):
        control_qubits = [control_qubits]
    if isinstance(target_qubits, (int, Qubit)):
        target_qubits = [target_qubits]

    if mode == 'no-ancilla':
        mcmt_gate = MCMTGateNoAncilla(gate, len(control_qubits), len(target_qubits))
        return self.append(mcmt_gate, control_qubits[:] + target_qubits[:], [])

    if mode == 'v-chain':
        mcmt_gate = MCMTGateVChain(gate, len(control_qubits), len(target_qubits))
        num_ancillas = mcmt_gate.num_ancilla_qubits
        ancilla_qubits = ancilla_qubits or []
        if isinstance(ancilla_qubits, (int, Qubit)):
            ancilla_qubits = [ancilla_qubits]
        elif len(ancilla_qubits) < num_ancillas:
            raise QiskitError('Insufficient number of ancilla qubits, need {}'.format(num_ancillas))

        return self.append(mcmt_gate,
                           control_qubits[:] + target_qubits[:] + ancilla_qubits[:num_ancillas],
                           [])

    raise QiskitError('Invalid mode specified.')


QuantumCircuit.mcmt = mcmt
