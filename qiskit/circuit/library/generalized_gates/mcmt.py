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

from __future__ import annotations

from qiskit.circuit import ControlledGate, Gate
from qiskit.circuit._utils import _ctrl_state_to_int
from ..standard_gates import get_standard_gate_name_mapping


class MCMTGate(ControlledGate):
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

    Depending on the number of available auxiliary qubits, this operation can be synthesized
    using different methods. For example, if :math:`n - 1` clean auxiliary qubits are available
    (where :math:`n` is the number of control qubits), a V-chain decomposition can be used whose
    depth is linear in :math:`n`. See also :func:`.synth_mcmt_chain`.
    """

    def __init__(
        self,
        gate: Gate,
        num_ctrl_qubits: int,
        num_target_qubits: int,
        ctrl_state: int | str | None = None,
        label: str | None = None,
    ) -> None:
        """
        Args:
            gate: The base gate to apply on multiple target qubits, controlled by other qubits.
                This must be a single-qubit gate or a controlled single-qubit gate.
            num_ctrl_qubits: The number of control qubits.
            num_target_qubits: The number of target qubits.
            ctrl_state: The control state of the control qubits. Defaults to all closed controls.
            label: The gate label.
        """
        if num_target_qubits < 1:
            raise ValueError("Need at least one target qubit.")

        if num_ctrl_qubits < 1:
            raise ValueError("Need at least one control qubit.")

        self.num_target_qubits = num_target_qubits

        base_gate = self._identify_base_gate(gate)
        num_qubits = num_ctrl_qubits + num_target_qubits

        if label is None:
            label = f"{num_target_qubits}-{gate.name.capitalize()}"

        super().__init__(
            "mcmt",
            base_gate=base_gate,
            num_qubits=num_qubits,
            params=gate.params,
            num_ctrl_qubits=num_ctrl_qubits,
            ctrl_state=ctrl_state,
            label=label,
        )

    def _define(self):
        """Default definition relying on gate.control. Control state is handled by superclass."""
        # pylint: disable=cyclic-import
        from qiskit.transpiler.passes.synthesis.hls_plugins import MCMTSynthesisDefault

        self.definition = MCMTSynthesisDefault().run(self)

    @staticmethod
    def _identify_base_gate(gate):
        """Get the control base gate. Note this must be a single qubit gate."""

        # try getting the standard name from the string
        if isinstance(gate, str):
            standard_gates = get_standard_gate_name_mapping()
            if gate in standard_gates:
                gate = standard_gates[gate]
            else:
                raise AttributeError(
                    f"Unknown gate {gate}. Available: {list(get_standard_gate_name_mapping.keys())}"
                )

        if not isinstance(gate, Gate):
            raise TypeError(f"Invalid gate type {type(gate)}.")

        if gate.num_qubits != 1:
            raise ValueError(
                f"MCMTGate requires a base gate with a single qubit, but got {gate.num_qubits}."
            )

        return gate

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None, annotated=False):
        """Return the controlled version of the MCMT circuit."""
        if not annotated:
            ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
            new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state

            gate = MCMTGate(
                self.base_gate,
                self.num_ctrl_qubits + num_ctrl_qubits,
                self.num_target_qubits,
                ctrl_state=new_ctrl_state,
            )
        else:
            gate = super().control(num_ctrl_qubits, label, ctrl_state, annotated=annotated)

        return gate

    def inverse(self, annotated: bool = False):
        """Return the inverse MCMT circuit."""
        return MCMTGate(
            self.base_gate.inverse(), self.num_ctrl_qubits, self.num_target_qubits, self.ctrl_state
        )
