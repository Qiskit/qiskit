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
from qiskit.circuit import Gate, Instruction, Qubit, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.extensions.standard.x import CCXGate
from qiskit.util import deprecate_arguments


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

    def __init__(self, controlled_gate: Union[Gate, Callable[[QuantumCircuit, Qubit, Qubit],
                                                             Instruction]],
                 num_ctrl_qubits: int,
                 num_target_qubits: int) -> None:
        """Create a new multi-control multi-target gate.

        Args:
            controlled_gate: The controlled gate to be applied controlled on the control qubits
                and applied to the target qubits. Can be either a Gate or a circuit method.
                If it is a callable, it will be casted to a Gate.
            num_ctrl_qubits: The number of control qubits.
            num_target_qubits: The number of target qubits.

        Raises:
            QiskitError: If the controlled gate does not act on exactly 2 qubits.
        """
        if callable(controlled_gate):
            q = QuantumRegister(2)
            circuit = QuantumCircuit(q)
            controlled_gate(circuit, q[0], q[1])
            controlled_gate = circuit.to_gate()

        if controlled_gate.num_qubits != 2:
            raise QiskitError('The target gate must be a single-qubit gate.')

        if num_ctrl_qubits == 0 or num_target_qubits == 0:
            raise QiskitError('Need at least one control and one target qubit.')

        self.controlled_gate = controlled_gate
        self.num_ctrl_qubits = num_ctrl_qubits
        self.num_target_qubits = num_target_qubits
        self.num_ancilla_qubits = max(0, num_ctrl_qubits - 1)

        num_qubits = num_ctrl_qubits + num_target_qubits + self.num_ancilla_qubits

        super().__init__('mcmt', num_qubits, [])

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return the controlled version of the MCMT gate."""
        if ctrl_state is None:
            return MCMTGate(self.controlled_gate,
                            self.num_ctrl_qubits + num_ctrl_qubits,
                            self.num_target_qubits)
        else:
            return super().control(num_ctrl_qubits, label, ctrl_state)

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
                (self.controlled_gate, [master_control, qubit], [])
            ]
        definition += _ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=True)

        self.definition = definition


# pylint:disable=unused-argument
@deprecate_arguments({'single_control_gate_fun': 'controlled_gate',
                      'q_controls': 'control_qubits',
                      'q_ancillae': 'ancilla_qubits',
                      'q_targets': 'target_qubits'})
def mcmt(self, controlled_gate, control_qubits, target_qubits, ancilla_qubits=None, mode=None,
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
                      'mcmct(controlled_gate, control_qubits, target_qubits, ancilla_qubits).',
                      DeprecationWarning, stacklevel=3)
        # swap arguments in the right order
        tmp = controlled_gate
        controlled_gate = target_qubits
        target_qubits = ancilla_qubits
        ancilla_qubits = control_qubits
        control_qubits = tmp

    if mode:
        warnings.warn('It seems you are using the old signature of the MCMT gate, which is '
                      'deprecated as of 0.13.0 and will be removed no earlier than 3 months '
                      'after the release date. The signature has been updated to '
                      'mcmct(controlled_gate, control_qubits, target_qubits, ancilla_qubits).'
                      'Setting the mode is no longer supported (since only one mode is available).',
                      DeprecationWarning, stacklevel=3)

    if isinstance(control_qubits, (int, Qubit)):
        control_qubits = [control_qubits]
    if isinstance(target_qubits, (int, Qubit)):
        target_qubits = [target_qubits]
    if isinstance(ancilla_qubits, (int, Qubit)):
        ancilla_qubits = [ancilla_qubits]

    gate = MCMTGate(controlled_gate, len(control_qubits), len(target_qubits))
    num_ancillas = gate.num_ancilla_qubits

    if num_ancillas == 0:
        ancilla_qubits = []
    elif ancilla_qubits is None:
        raise QiskitError('{} ancilla qubits are needed but None were given.'.format(num_ancillas))
    elif len(ancilla_qubits) < num_ancillas:
        raise QiskitError('Insufficient number of ancilla qubits, need {}'.format(num_ancillas))

    return self.append(gate,
                       control_qubits[:] + target_qubits[:] + ancilla_qubits[:num_ancillas],
                       [])


QuantumCircuit.mcmt = mcmt
