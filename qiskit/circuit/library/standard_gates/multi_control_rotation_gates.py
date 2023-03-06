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
from qiskit.circuit.library.generalized_gates import MCU2Gate
from qiskit.circuit.library.standard_gates import XGate, YGate, ZGate, RXGate, RYGate, RZGate
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit._utils import _ctrl_state_to_int, _compute_control_matrix


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

    n_c = len(control_qubits)
    if n_c <= 6:
        ncrx = ControlRotationGate(theta, n_c, axis="x")
    else:
        rxgate = numpy.cos(theta / 2) * numpy.eye(2) - 1j * numpy.sin(theta / 2) * XGate().__array__()
        ncrx = MCU2Gate(rxgate,
                        n_c,
                        label=f"Rx({theta:0.3f})")

    # if use_basis_gates:
    #     ncrx = transpile(ncrx, basis_gates=['cx','u', 'p'])

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

    n_c = len(control_qubits)
    if n_c <= 6:
        ncry = ControlRotationGate(theta, n_c, axis="y")
    else:
        rygate = numpy.cos(theta / 2) * numpy.eye(2) - 1j * numpy.sin(theta / 2) * YGate().__array__()
        ncry = MCU2Gate(rygate,
                        n_c,
                        label=f"Ry({theta:0.3f})")

    # if use_basis_gates:
    #     ncry = transpile(ncry, basis_gates=['cx','u', 'p'])

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

    n_c = len(control_qubits)
    if n_c <= 6:
        ncrz = ControlRotationGate(lam, n_c, axis="z")
    else:
        rzgate = numpy.cos(lam / 2) * numpy.eye(2) - 1j * numpy.sin(lam / 2) * ZGate().__array__()
        ncrz = MCU2Gate(rzgate,
                        n_c,
                        label=f"Rz({lam:0.3f})")
    # if use_basis_gates:
    #     ncrz = transpile(ncrz, basis_gates=['cx','u', 'p'])

    self.append(ncrz, [*control_qubits, q_target])


QuantumCircuit.mcrx = mcrx
QuantumCircuit.mcry = mcry
QuantumCircuit.mcrz = mcrz


class ControlRotationGate(ControlledGate):
    """
    Control rotation gate. See Theorem 8 of https://arxiv.org/pdf/quant-ph/0406176.pdf

    Generate quantum circuit implementing a rotation generated by a single qubit Pauli operator.

    Args:
        angle (float): angle of rotation
        ctrl_list (list): list of control qubit indices
        target (int): index of target qubit
        axis (str): x,y,z axis
    Returns:
        circ (QuantumCircuit): quantum circuit implementing multicontrol rotation

    """

    def __init__(self, angle: float,
                 num_ctrl_qubits: int,
                 axis: str,
                 ctrl_state: Optional[Union[str, int]] = None):

        assert axis in ["x", "y", "z"], f"can only rotated around x,y,z axis, not {axis}"

        self.axis = axis
        self.angle = angle

        if self.axis == 'x':
            self.base_gate = RXGate(angle)
        elif self.axis == 'y':
            self.base_gate = RYGate(angle)
        elif self.axis == 'z':
            self.base_gate = RZGate(angle)

        self._num_qubits = num_ctrl_qubits + 1
        self.num_ctrl_qubits = num_ctrl_qubits
        self.ctrl_state = ctrl_state

        self.label = self.base_gate.label
        super().__init__(
            name=self.label,
            num_qubits=self._num_qubits,
            params=[angle],
            num_ctrl_qubits=self.num_ctrl_qubits,
            ctrl_state=self.ctrl_state,
            base_gate=self.base_gate,
        )

    def _define(self):

        cntrl_int = _ctrl_state_to_int(self.ctrl_state, self.num_ctrl_qubits)
        cntrl_str = numpy.binary_repr(cntrl_int, width=self.num_ctrl_qubits)[::-1]

        target_qbit = self.num_ctrl_qubits
        rot_circuit = custom_mcrtl_rot(self.angle,
                                       list(range(self.num_ctrl_qubits)),
                                       target_qbit,
                                       self.axis)

        controls = QuantumRegister(self.num_ctrl_qubits)
        target = QuantumRegister(1)

        control_circ = QuantumCircuit(controls, target)
        for q_ind, cntrol_bit in enumerate(cntrl_str):
            if cntrol_bit == '0':
                control_circ.x(q_ind)
        self.definition = control_circ.compose(rot_circuit).compose(control_circ)

    def inverse(self):
        """
        Returns inverse rotation gate
        """
        return ControlRotationGate(-1 * self.angle,
                                   self.num_ctrl_qubits,
                                   self.axis,
                                   ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """
        Return numpy array for gate
        """
        mat = _compute_control_matrix(
            self.base_gate.to_matrix(),
            self.num_ctrl_qubits,
            ctrl_state=self.ctrl_state)
        if dtype:
            mat = numpy.asarray(mat, dtype=dtype)
        return mat
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
