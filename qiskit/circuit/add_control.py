# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Add control to operation if supported.
"""
from typing import Union, Optional

from qiskit.circuit.exceptions import CircuitError
from qiskit.extensions import UnitaryGate
from qiskit.extensions.standard import XGate, RXGate, RYGate, RZGate, U1Gate, U3Gate
from . import ControlledGate, Gate, QuantumRegister, QuantumCircuit


def add_control(operation: Union[Gate, ControlledGate],
                num_ctrl_qubits: int,
                label: Union[str, None],
                ctrl_state: Union[int, str, None]) -> ControlledGate:
    """For standard gates, if the controlled version already exists in the
    library, it will be returned (e.g. XGate.control() = CnotGate().

    For more generic gates, this method implements the controlled
    version by first decomposing into the ['u1', 'u3', 'cx'] basis, then
    controlling each gate in the decomposition.

    Open controls are implemented by conjugating the control line with
    X gates. Adds num_ctrl_qubits controls to operation.

    Args:
        operation: Operation for which control will be added.
        num_ctrl_qubits: The number of controls to add to gate (default=1).
        label: Optional gate label.
        ctrl_state (int or str or None): The control state in decimal or as
            a bitstring (e.g. '111'). If specified as a bitstring the length
            must equal num_ctrl_qubits, MSB on left. If None, use
            2**num_ctrl_qubits-1.

    Returns:
        Controlled version of gate.

    """
    import qiskit.extensions.standard as standard
    if isinstance(operation, standard.RZGate) or operation.name == 'rz':
        # num_ctrl_qubits > 1
        # the condition matching 'name' above is to catch a test case,
        # 'TestControlledGate.test_rotation_gates', where the rz gate
        # gets converted to a circuit before becoming a generic Gate object.
        cgate = standard.CRZGate(*operation.params)
        return cgate.control(num_ctrl_qubits - 1)
    if isinstance(operation, UnitaryGate):
        # attempt decomposition
        operation._define()
    return control(operation, num_ctrl_qubits=num_ctrl_qubits, label=label,
                   ctrl_state=ctrl_state)


def control(operation: Union[Gate, ControlledGate],
            num_ctrl_qubits: Optional[int] = 1,
            label: Optional[Union[None, str]] = None,
            ctrl_state: Optional[Union[None, int, str]] = None) -> ControlledGate:
    """Return controlled version of gate using controlled rotations

    Args:
        operation: gate to create ControlledGate from
        num_ctrl_qubits: number of controls to add to gate (default=1)
        label: optional gate label
        ctrl_state: The control state in decimal or as
            a bitstring (e.g. '111'). If specified as a bitstring the length
            must equal num_ctrl_qubits, MSB on left. If None, use
            2**num_ctrl_qubits-1.

    Returns:
        Controlled version of gate.

    Raises:
        CircuitError: gate contains non-gate in definition
    """
    # pylint: disable=cyclic-import
    import qiskit.circuit.controlledgate as controlledgate
    # pylint: disable=unused-import
    import qiskit.extensions.standard.multi_control_rotation_gates
    import qiskit.extensions.standard.multi_control_toffoli_gate
    import qiskit.extensions.standard.multi_control_u1_gate

    q_control = QuantumRegister(num_ctrl_qubits, name='control')
    q_target = QuantumRegister(operation.num_qubits, name='target')
    q_ancillae = None  # TODO: add
    qc = QuantumCircuit(q_control, q_target)

    qubit_kwargs = {
        'q_controls': q_control[:] + q_target[:-1],
        'q_target': q_target[-1],
    }

    if _operation_has_base_gate(operation, XGate):
        qc.mct(**qubit_kwargs,
               q_ancilla=q_ancillae,
               mode='noancilla')
    elif _operation_has_base_gate(operation, RXGate):
        qc.mcrx(_get_base_gate_params(operation)[0], **qubit_kwargs,
                use_basis_gates=True)
    elif _operation_has_base_gate(operation, RYGate):
        qc.mcry(_get_base_gate_params(operation)[0], **qubit_kwargs,
                q_ancillae=q_ancillae, use_basis_gates=True)
    elif isinstance(operation, RZGate):
        qc.mcrz(operation.definition[0][0].params[0], **qubit_kwargs,
                use_basis_gates=True)
    elif _operation_has_base_gate(operation, U1Gate):
        qc.mcu1(_get_base_gate_params(operation)[2], *qubit_kwargs.values())
    elif _operation_has_base_gate(operation, U3Gate):
        theta, phi, lamb = _get_base_gate_params(operation)
        _apply_mcu3(qc, theta, phi, lamb, **qubit_kwargs, q_ancillae=q_ancillae)
    else:
        bgate = _unroll_gate(operation, ['u1', 'u3', 'cx'])
        # now we have a bunch of single qubit rotation gates and cx
        for rule in bgate.definition:
            if rule[0].name == 'u3':
                theta, phi, lamb = rule[0].params
                _apply_mcu3(qc, theta, phi, lamb, q_control, q_target[rule[1][0].index],
                            q_ancillae)
            elif rule[0].name == 'u1':
                qc.mcu1(rule[0].params[0], q_control, q_target[rule[1][0].index])
            elif rule[0].name == 'cx':
                qc.mct(q_control[:] + [q_target[rule[1][0].index]],
                       q_target[rule[1][1].index],
                       None,
                       mode='noancilla')
            else:
                raise CircuitError('gate contains non-controllable instructions')
    instr = qc.to_instruction()
    if isinstance(operation, controlledgate.ControlledGate):
        new_num_ctrl_qubits = num_ctrl_qubits + operation.num_ctrl_qubits
        base_name = operation.base_gate.name
        base_gate = operation.base_gate
    else:
        new_num_ctrl_qubits = num_ctrl_qubits
        base_name = operation.name
        base_gate = operation
    # In order to maintain some backward compatibility with gate names this
    # uses a naming convention where if the number of controls is <=2 the gate
    # is named like "cc<base_gate.name>", else it is named like
    # "c<num_ctrl_qubits><base_name>".
    if new_num_ctrl_qubits > 2:
        ctrl_substr = 'c{0:d}'.format(new_num_ctrl_qubits)
    else:
        ctrl_substr = ('{0}' * new_num_ctrl_qubits).format('c')
    new_name = '{0}{1}'.format(ctrl_substr, base_name)
    cgate = controlledgate.ControlledGate(new_name,
                                          instr.num_qubits,
                                          operation.params,
                                          label=label,
                                          num_ctrl_qubits=new_num_ctrl_qubits,
                                          definition=instr.definition,
                                          ctrl_state=ctrl_state)
    cgate.base_gate = base_gate
    return cgate


def _gate_to_circuit(operation):
    qr = QuantumRegister(operation.num_qubits)
    qc = QuantumCircuit(qr, name=operation.name)
    if hasattr(operation, 'definition') and operation.definition:
        for rule in operation.definition:
            if rule[0].name in {'id', 'barrier', 'measure', 'snapshot'}:
                raise CircuitError('Cannot make controlled gate with {} instruction'.format(
                    rule[0].name))
            qc.append(rule[0], qargs=[qr[bit.index] for bit in rule[1]], cargs=[])
    else:
        qc.append(operation, qargs=qr, cargs=[])
    return qc


def _operation_has_base_gate(operation, gate):
    return isinstance(operation, gate) or (
        isinstance(operation, ControlledGate) and
        isinstance(operation.base_gate, gate))


def _get_base_gate_params(operation):
    if isinstance(operation, ControlledGate):
        if isinstance(operation.base_gate, U3Gate):
            return operation.base_gate.params
        else:
            return operation.base_gate.definition[0][0].params
    else:
        if isinstance(operation, U3Gate):
            return operation.params
        else:
            return operation.definition[0][0].params


def _unroll_gate(operation, basis_gates):
    from qiskit.converters.circuit_to_dag import circuit_to_dag
    from qiskit.converters.dag_to_circuit import dag_to_circuit
    from qiskit.transpiler.passes import Unroller
    unroller = Unroller(basis_gates)
    dag = circuit_to_dag(_gate_to_circuit(operation))
    qc = dag_to_circuit(unroller.run(dag))
    return qc.to_gate()


def _apply_mcu3(circuit, theta, phi, lamb, q_controls, q_target, q_ancillae):
    from math import pi

    if phi == -pi / 2 and lamb == pi / 2:
        circuit.mcrx(theta, q_controls, q_target, use_basis_gates=True)
    elif phi == 0 and lamb == 0:
        circuit.mcry(theta, q_controls, q_target,
                     q_ancillae, mode='noancilla', use_basis_gates=True)
    elif theta == 0 and phi == 0:
        circuit.mcrz(lamb, q_controls, q_target, use_basis_gates=True)
    else:
        circuit.mcrz(lamb, q_controls, q_target, use_basis_gates=True)
        circuit.mcry(theta, q_controls, q_target,
                     q_ancillae, mode='noancilla', use_basis_gates=True)
        circuit.mcrz(phi, q_controls, q_target, use_basis_gates=True)
