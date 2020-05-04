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
from qiskit.circuit import ControlledGate, Gate, Instruction, Qubit, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.util import deprecate_arguments
from ..standard_gates import (
    XGate, YGate, ZGate, HGate, TGate, TdgGate, SGate, SdgGate
)


class MCMT(QuantumCircuit):
    """The multi-controlled multi-target gate, for an arbitrary singly controlled target gate.

    For example, the H gate controlled on 3 qubits and acting on 2 target qubit is represented as:

    .. parsed-literal::

        ───■────
           │
        ───■────
           │
        ───■────
        ┌──┴───┐
        ┤0     ├
        │  2-H │
        ┤1     ├
        └──────┘

    This default implementations requires no ancilla qubits, by broadcasting the target gate
    to the number of target qubits and using Qiskit's generic control routine to control the
    broadcasted target on the control qubits. If ancilla qubits are available, a more efficient
    variant using the so-called V-chain decomposition can be used. This is implemented in
    :class:`~qiskit.circuit.library.MCMTVChain`.
    """

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
            AttributeError: If the gate cannot be casted to a controlled gate.
            AttributeError: If the number of controls or targets is 0.
        """
        if num_ctrl_qubits == 0 or num_target_qubits == 0:
            raise AttributeError('Need at least one control and one target qubit.')

        # set the internal properties and determine the number of qubits
        self.gate = self._identify_gate(gate)
        self.num_ctrl_qubits = num_ctrl_qubits
        self.num_target_qubits = num_target_qubits
        num_qubits = num_ctrl_qubits + num_target_qubits + self.num_ancilla_qubits

        # initialize the circuit object
        super().__init__(num_qubits, name='mcmt')

        # build the circuit
        self._build()

    def _build(self):
        """Define the MCMT gate without ancillas."""
        if self.num_target_qubits == 1:
            # no broadcasting needed (makes for better circuit diagrams)
            broadcasted_gate = self.gate
        else:
            name = '{}-{}'.format(self.num_target_qubits, self.gate.name.capitalize())
            broadcasted = QuantumCircuit(self.num_target_qubits, name=name)
            for target in list(range(self.num_target_qubits)):
                broadcasted.append(self.gate, [target], [])
            broadcasted_gate = broadcasted.to_gate()

        mcmt_gate = broadcasted_gate.control(self.num_ctrl_qubits)
        self.append(mcmt_gate, self.qubits, [])

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancillas."""
        return 0

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
        elif isinstance(gate, QuantumCircuit):
            if gate.num_qubits != 1:
                raise AttributeError('The circuit you specified as control gate can only have '
                                     'one qubit!')
            base_gate = gate.to_gate()  # raises error if circuit contains non-unitary instructions
        else:
            if callable(gate):  # identify via name of the passed function
                name = gate.__name__
            elif isinstance(gate, str):
                name = gate
            else:
                raise AttributeError('Invalid gate specified: {}'.format(gate))
            base_gate = valid_gates[name]

        return base_gate

    def control(self, num_ctrl_qubits=1):
        """Return the controlled version of the MCMT circuit."""
        return MCMT(self.gate,
                    self.num_ctrl_qubits + num_ctrl_qubits,
                    self.num_target_qubits)

    def inverse(self):
        """Return the inverse MCMT circuit, which is itself."""
        return MCMT(self.gate, self.num_ctrl_qubits, self.num_target_qubits)


class MCMTVChain(MCMT):
    """The MCMT implementation using the CCX V-chain.

    This implementation requires ancillas but is decomposed into a much shallower circuit
    than the default implementation in :class:`~qiskit.circuit.library.MCMT`.

    **Expanded Circuit:**

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import MCMTVChain, ZGate
        import qiskit.tools.jupyter
        circuit = MCMTVChain(ZGate(), 2, 2)
        %circuit_library_info circuit.decompose()

    **Examples:**

        >>> from qiskit.circuit.library import HGate
        >>> MCMTVChain(HGate(), 3, 2).draw()

        q_0: ──■────────────────────────■──
               │                        │
        q_1: ──■────────────────────────■──
               │                        │
        q_2: ──┼────■──────────────■────┼──
               │    │  ┌───┐       │    │
        q_3: ──┼────┼──┤ H ├───────┼────┼──
               │    │  └─┬─┘┌───┐  │    │
        q_4: ──┼────┼────┼──┤ H ├──┼────┼──
             ┌─┴─┐  │    │  └─┬─┘  │  ┌─┴─┐
        q_5: ┤ X ├──■────┼────┼────■──┤ X ├
             └───┘┌─┴─┐  │    │  ┌─┴─┐└───┘
        q_6: ─────┤ X ├──■────■──┤ X ├─────
                  └───┘          └───┘
    """

    def _build(self):
        """Define the MCMT gate."""
        control_qubits = self.qubits[:self.num_ctrl_qubits]
        target_qubits = self.qubits[self.num_ctrl_qubits:
                                    self.num_ctrl_qubits + self.num_target_qubits]
        ancilla_qubits = self.qubits[self.num_ctrl_qubits + self.num_target_qubits:]

        if len(ancilla_qubits) > 0:
            master_control = ancilla_qubits[-1]
        else:
            master_control = control_qubits[0]

        self._ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=False)
        for qubit in target_qubits:
            self.append(self.gate.control(), [master_control, qubit], [])
        self._ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=True)

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancilla qubits required."""
        return max(0, self.num_ctrl_qubits - 1)

    def _ccx_v_chain_rule(self, control_qubits: Union[QuantumRegister, List[Qubit]],
                          ancilla_qubits: Union[QuantumRegister, List[Qubit]],
                          reverse: bool = False) -> List[Tuple[Gate, List[Qubit], List]]:
        """Get the rule for the CCX V-chain.

        The CCX V-chain progressively computes the CCX of the control qubits and puts the final
        result in the last ancillary qubit.

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
            return

        if len(ancilla_qubits) < len(control_qubits) - 1:
            raise QiskitError('Insufficient number of ancilla qubits.')

        iterations = list(enumerate(range(2, len(control_qubits))))
        if not reverse:
            self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])
            for i, j in iterations:
                self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
        else:
            for i, j in reversed(iterations):
                self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
            self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])

    def inverse(self):
        return MCMTVChain(self.gate, self.num_ctrl_qubits, self.num_target_qubits)


# pylint:disable=unused-argument
@deprecate_arguments({'single_control_gate_fun': 'gate',
                      'q_controls': 'control_qubits',
                      'q_ancillae': 'ancilla_qubits',
                      'q_targets': 'target_qubits'})
def mcmt(self, gate, control_qubits, target_qubits, ancilla_qubits=None, mode='noancilla',
         *, single_control_gate_fun=None, q_controls=None, q_ancillae=None, q_targets=None):
    """Apply a multi-control, multi-target using a generic gate.

    This can also be used to implement a generic multi-control gate, as the target could also be of
    length 1.
    """
    warnings.warn('The multi-control multi-target gate has moved to the circuit library as of '
                  '0.14.0 and will not be useable as circuit method anymore. This method will be '
                  'removed no earlier than 3 months after the release date. '
                  'You should create the qiskit.circuit.library.MCMT or MCMTVChain circuits and '
                  'add then to your circuit using append, extend, or compose.',
                  DeprecationWarning, stacklevel=3)

    deprecated_modes = {'no-ancilla': 'noancilla',
                        'basic': 'v-chain'}

    if mode in deprecated_modes.keys():
        warnings.warn('The mode supplied mode for ``QuantumCircuit.mcmt`` is deprecated, use '
                      '{} instead of {}.'.format(deprecated_modes[mode], mode))
        mode = deprecated_modes[mode]

    # for backward compatibility; the previous signature was
    # `def mcmt(self, q_controls, q_ancillae, single_control_gate_fun, q_targets, mode="basic")`
    if callable(target_qubits):
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

    if mode == 'noancilla':
        mcmt_gate = MCMT(gate, len(control_qubits), len(target_qubits))
        return self.append(mcmt_gate, control_qubits[:] + target_qubits[:], [])

    if mode == 'v-chain':
        mcmt_gate = MCMTVChain(gate, len(control_qubits), len(target_qubits))
        num_ancillas = mcmt_gate.num_ancilla_qubits
        ancilla_qubits = ancilla_qubits or []
        if isinstance(ancilla_qubits, (int, Qubit)):
            ancilla_qubits = [ancilla_qubits]
        elif len(ancilla_qubits) < num_ancillas:
            raise QiskitError('Insufficient number of ancilla qubits, need {}'.format(num_ancillas))

        return self.append(mcmt_gate,
                           control_qubits[:] + target_qubits[:] + ancilla_qubits[:num_ancillas],
                           [])

    raise QiskitError('Invalid mode specified: {}'.format(mode))


QuantumCircuit.mcmt = mcmt
