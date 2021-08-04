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
Multiple-Controlled U3 gate. Not using ancillary qubits.
"""

from math import pi
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.exceptions import QiskitError


def _apply_cu(circuit, theta, phi, lam, control, target, use_basis_gates=True):
    if use_basis_gates:
        # pylint: disable=cyclic-import
        circuit.p((lam + phi) / 2, [control])
        circuit.p((lam - phi) / 2, [target])
        circuit.cx(control, target)
        circuit.u(-theta / 2, 0, -(phi + lam) / 2, [target])
        circuit.cx(control, target)
        circuit.u(theta / 2, phi, 0, [target])
    else:
        circuit.cu(theta, phi, lam, 0, control, target)


def _apply_mcu_graycode(circuit, theta, phi, lam, ctls, tgt, use_basis_gates):
    """Apply multi-controlled u gate from ctls to tgt using graycode
    pattern with single-step angles theta, phi, lam."""

    n = len(ctls)

    gray_code = _generate_gray_code(n)
    last_pattern = None

    for pattern in gray_code:
        if "1" not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # find left most set bit
        lm_pos = list(pattern).index("1")

        # find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == "1"]
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        # check parity and undo rotation
        if pattern.count("1") % 2 == 0:
            # inverse CU: u(theta, phi, lamb)^dagger = u(-theta, -lam, -phi)
            _apply_cu(
                circuit, -theta, -lam, -phi, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates
            )
        else:
            _apply_cu(circuit, theta, phi, lam, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates)
        last_pattern = pattern


def mcrx(self, theta, q_controls, q_target, use_basis_gates=False):
    """
    Apply Multiple-Controlled X rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrx gate on.
        theta (float): angle theta
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """

    # check controls
    if isinstance(q_controls, QuantumRegister):
        control_qubits = list(q_controls)
    elif isinstance(q_controls, list):
        control_qubits = q_controls
    else:
        raise QiskitError(
            "The mcrx gate needs a list of qubits or a quantum register for controls."
        )

    # check target
    if isinstance(q_target, Qubit):
        target_qubit = q_target
    else:
        raise QiskitError("The mcrx gate needs a single qubit as target.")

    all_qubits = control_qubits + [target_qubit]

    self._check_qargs(all_qubits)
    self._check_dups(all_qubits)

    n_c = len(control_qubits)
    if n_c == 1:  # cu
        _apply_cu(
            self,
            theta,
            -pi / 2,
            pi / 2,
            control_qubits[0],
            target_qubit,
            use_basis_gates=use_basis_gates,
        )
    else:
        theta_step = theta * (1 / (2 ** (n_c - 1)))
        _apply_mcu_graycode(
            self,
            theta_step,
            -pi / 2,
            pi / 2,
            control_qubits,
            target_qubit,
            use_basis_gates=use_basis_gates,
        )


def mcry(self, theta, q_controls, q_target, q_ancillae=None, mode=None, use_basis_gates=False):
    """
    Apply Multiple-Controlled Y rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcry gate on.
        theta (float): angle theta
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        q_ancillae (QuantumRegister or tuple(QuantumRegister, int)): The list of ancillary qubits.
        mode (string): The implementation mode to use
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """

    # check controls
    if isinstance(q_controls, QuantumRegister):
        control_qubits = list(q_controls)
    elif isinstance(q_controls, list):
        control_qubits = q_controls
    else:
        raise QiskitError(
            "The mcry gate needs a list of qubits or a quantum " "register for controls."
        )

    # check target
    if isinstance(q_target, Qubit):
        target_qubit = q_target
    else:
        raise QiskitError("The mcry gate needs a single qubit as target.")

    # check ancilla
    if q_ancillae is None:
        ancillary_qubits = []
    elif isinstance(q_ancillae, QuantumRegister):
        ancillary_qubits = list(q_ancillae)
    elif isinstance(q_ancillae, list):
        ancillary_qubits = q_ancillae
    else:
        raise QiskitError(
            "The mcry gate needs None or a list of qubits or a " "quantum register for ancilla."
        )

    all_qubits = control_qubits + [target_qubit] + ancillary_qubits

    self._check_qargs(all_qubits)
    self._check_dups(all_qubits)

    # auto-select the best mode
    if mode is None:
        # if enough ancillary qubits are provided, use the 'v-chain' method
        additional_vchain = MCXGate.get_num_ancilla_qubits(len(control_qubits), "v-chain")
        if len(ancillary_qubits) >= additional_vchain:
            mode = "basic"
        else:
            mode = "noancilla"

    if mode == "basic":
        self.ry(theta / 2, q_target)
        self.mcx(q_controls, q_target, q_ancillae, mode="v-chain")
        self.ry(-theta / 2, q_target)
        self.mcx(q_controls, q_target, q_ancillae, mode="v-chain")
    elif mode == "noancilla":
        n_c = len(control_qubits)
        if n_c == 1:  # cu
            _apply_cu(
                self, theta, 0, 0, control_qubits[0], target_qubit, use_basis_gates=use_basis_gates
            )
        else:
            theta_step = theta * (1 / (2 ** (n_c - 1)))
            _apply_mcu_graycode(
                self,
                theta_step,
                0,
                0,
                control_qubits,
                target_qubit,
                use_basis_gates=use_basis_gates,
            )
    else:
        raise QiskitError(f"Unrecognized mode for building MCRY circuit: {mode}.")


def mcrz(self, lam, q_controls, q_target, use_basis_gates=False):
    """
    Apply Multiple-Controlled Z rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrz gate on.
        lam (float): angle lambda
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """

    # check controls
    if isinstance(q_controls, QuantumRegister):
        control_qubits = list(q_controls)
    elif isinstance(q_controls, list):
        control_qubits = q_controls
    else:
        raise QiskitError(
            "The mcrz gate needs a list of qubits or a quantum register for controls."
        )

    # check target
    if isinstance(q_target, Qubit):
        target_qubit = q_target
    else:
        raise QiskitError("The mcrz gate needs a single qubit as target.")

    all_qubits = control_qubits + [target_qubit]

    self._check_qargs(all_qubits)
    self._check_dups(all_qubits)

    n_c = len(control_qubits)
    if n_c == 1:  # cu
        _apply_cu(self, 0, 0, lam, control_qubits[0], target_qubit, use_basis_gates=use_basis_gates)
    else:
        lam_step = lam * (1 / (2 ** (n_c - 1)))
        _apply_mcu_graycode(
            self, 0, 0, lam_step, control_qubits, target_qubit, use_basis_gates=use_basis_gates
        )


QuantumCircuit.mcrx = mcrx
QuantumCircuit.mcry = mcry
QuantumCircuit.mcrz = mcrz
