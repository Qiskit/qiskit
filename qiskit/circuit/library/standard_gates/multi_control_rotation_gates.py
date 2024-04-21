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

from typing import Optional, Union, Tuple, List

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.utils.deprecation import deprecate_func


@deprecate_func(since="1.1.0", additional_msg="Use QuantumCircuit.mcrx instead.")
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
    return self.mcrx(theta, q_controls, q_target, use_basis_gates, use_basis_gates)


@deprecate_func(since="1.1.0", additional_msg="Use QuantumCircuit.mcry instead.")
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
    return self.mcry(theta, q_controls, q_target, q_ancillae, mode, use_basis_gates)


@deprecate_func(since="1.1.0", additional_msg="Use QuantumCircuit.mcrz instead.")
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
    return self.mcrz(lam, q_controls, q_target, use_basis_gates, use_basis_gates)
