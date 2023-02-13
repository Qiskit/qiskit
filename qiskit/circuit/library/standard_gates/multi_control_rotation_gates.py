# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""multicontrol rotation gates around an axis in x,y and z planes."""

from typing import Optional, Union, Tuple, List
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
from qiskit.circuit.library.generalized_gates import multiconrol_single_qubit_gate
from qiskit.circuit.library.standard_gates import XGate, YGate, ZGate
from numpy import eye, cos, sin

I = eye(2)
Z = ZGate().__array__()  # array([[1, 0], [0, -1]])
X = XGate().__array__()  # array([[0, 1], [1, 0]])
Y = YGate().__array__()  # array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]])


def mcrx(
    self,
    theta: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    use_basis_gates: bool = False,
):
    """
    Apply Multiple-Controlled X rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrx gate on.
        theta (float): angle theta
        q_controls (QuantumRegister or list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    all_qubits = control_qubits + target_qubit
    self._check_dups(all_qubits)

    contrl_qubits = list(range(len(control_qubits)))
    targ_qubit = contrl_qubits[-1] + 1
    n_c = len(control_qubits)

    if n_c <= 6:
        ncrx = custom_mcrtl_rot(theta, contrl_qubits, targ_qubit, axis="x")
    else:
        rxgate = cos(theta / 2) * I - 1j * sin(theta / 2) * X
        ncrx = multiconrol_single_qubit_gate(rxgate, contrl_qubits, targ_qubit)

    # if use_basis_gates:
    #     ncrx = transpile(ncrx, basis_gates=['cx','u', 'p'])

    ncrx.name = f"MC-Rx({theta:0.3f})"
    self.append(ncrx, [*control_qubits, q_target])


def mcry(
    self,
    theta: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    q_ancillae: Optional[Union[QuantumRegister, Tuple[QuantumRegister, int]]] = None,
    mode: str = None,
    use_basis_gates=False,
):
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
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    ancillary_qubits = [] if q_ancillae is None else self.qbit_argument_conversion(q_ancillae)
    all_qubits = control_qubits + target_qubit + ancillary_qubits
    self._check_dups(all_qubits)

    contrl_qubits = list(range(len(control_qubits)))
    targ_qubit = contrl_qubits[-1] + 1
    n_c = len(control_qubits)

    if n_c <= 6:
        ncry = custom_mcrtl_rot(theta, contrl_qubits, targ_qubit, axis="y")
    else:
        rygate = cos(theta / 2) * I - 1j * sin(theta / 2) * Y
        ncry = multiconrol_single_qubit_gate(rygate, contrl_qubits, targ_qubit)

    # if use_basis_gates:
    #     ncry = transpile(ncry, basis_gates=['cx','u', 'p'])

    ncry.name = f"MC-Ry({theta:0.3f})"
    self.append(ncry, [*control_qubits, q_target])


def mcrz(
    self,
    lam: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    use_basis_gates: bool = False,
):
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
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)

    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    all_qubits = control_qubits + target_qubit
    self._check_dups(all_qubits)

    contrl_qubits = list(range(len(control_qubits)))
    targ_qubit = contrl_qubits[-1] + 1
    n_c = len(control_qubits)

    if n_c <= 6:
        ncrz = custom_mcrtl_rot(lam, contrl_qubits, targ_qubit, axis="z")
    else:
        rzgate = cos(lam / 2) * I - 1j * sin(lam / 2) * Z
        ncrz = multiconrol_single_qubit_gate(rzgate, contrl_qubits, targ_qubit)

    # if use_basis_gates:
    #     ncrz = transpile(ncrz, basis_gates=['cx','u', 'p'])

    ncrz.name = f"MC-Rz({lam:0.3f})"
    self.append(ncrz, [*control_qubits, q_target])


QuantumCircuit.mcrx = mcrx
QuantumCircuit.mcry = mcry
QuantumCircuit.mcrz = mcrz


def custom_mcrtl_rot(
    angle: float, ctrl_list: List[int], target: int, axis: str = "y"
) -> QuantumCircuit:
    """
    See Theorem 8 of https://arxiv.org/pdf/quant-ph/0406176.pdf

    Generate quantum circuit implementing a rotation generated by a single qubit Pauli operator.

    Args:
        angle (float): angle of rotation
        ctrl_list (list): list of control qubit indices
        target (int): index of target qubit
        axis (str): x,y,z axis
    Returns:
        circ (QuantumCircuit): quantum circuit implementing multicontrol rotation
    """
    assert axis in ["x", "y", "z"], f"can only rotated around x,y,z axis, not {axis}"
    assert target not in ctrl_list, f"target qubit: {target} in control list"

    ctrl_list = sorted(ctrl_list)
    n_ctrl = len(ctrl_list)
    circ = QuantumCircuit(max([target, max(ctrl_list)]) + 1)
    pattern = _get_pattern(n_ctrl)

    if n_ctrl > 1:
        for s, i in enumerate(pattern):
            j = ctrl_list[-i - 1]
            p = _primitive_block((-1) ** s * angle, n_ctrl, axis)
            circ = circ.compose(p, [j, target])

        circ = circ.compose(circ)
    else:
        p_plus = _primitive_block(+angle, n_ctrl, axis)
        p_minus = _primitive_block(-angle, n_ctrl, axis)

        circ = circ.compose(p_plus, [ctrl_list[0], target])
        circ = circ.compose(p_minus, [ctrl_list[0], target])

    return circ


def _get_pattern(n_ctrl: int, _pattern=[0]) -> List[int]:
    """
    Recursively get list of control indices for multicontrol rotation gate

    Args:
        n_ctrl (int): number of controls
        _pattern (list): stores list of control indices in each recursive call. This should not be changed by user
    Returns
        list of control indices

    """
    if n_ctrl == _pattern[-1] + 1:
        return _pattern
    else:
        new_pattern = _pattern * 2
        new_pattern[-1] += 1
        return _get_pattern(n_ctrl, new_pattern)


def _primitive_block(angle: float, n_ctrl: int, axis: str) -> QuantumCircuit:
    """
    Get primative gates to perform multicontrol rotation

    Args:
        angle (float): angle of rotation
        n_ctrl (int): number of control qubits
        axis (str): axis of rotation (x,y or z)
    Returns:
        primitive (QuantumCircuit): quantum circuit of primative needed to perform multicontrol rotation
    """
    primitive = QuantumCircuit(2)
    if axis == "x":
        primitive.rx(angle / (2**n_ctrl), 1)
        primitive.cz(0, 1)
    elif axis == "y":
        primitive.ry(angle / (2**n_ctrl), 1)
        primitive.cx(0, 1)
    elif axis == "z":
        primitive.rz(angle / (2**n_ctrl), 1)
        primitive.cx(0, 1)
    else:
        raise ValueError("Unrecognised axis, must be one of x,y or z.")

    return primitive
