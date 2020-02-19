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
from qiskit import QiskitError
from qiskit.extensions import UnitaryGate


def add_control(operation, num_ctrl_qubits, label):
    """Add num_ctrl_qubits controls to operation

    Args:
        operation (Gate or ControlledGate): operation to add control to.
        num_ctrl_qubits (int): number of controls to add to gate (default=1)
        label (str): optional gate label

    Returns:
        ControlledGate: controlled version of gate. This default algorithm
            uses num_ctrl_qubits-1 ancillae qubits so returns a gate of size
            num_qubits + 2*num_ctrl_qubits - 1.
    """
    import qiskit.extensions.standard as standard
    if isinstance(operation, standard.RZGate) or operation.name == 'rz':
        # num_ctrl_qubits > 1
        # the condition matching 'name' above is to catch a test case,
        # 'TestControlledGate.test_rotation_gates', where the rz gate
        # gets converted to a circuit before becoming a generic Gate object.
        cgate = standard.CrzGate(*operation.params)
        return cgate.control(num_ctrl_qubits - 1)
    if isinstance(operation, UnitaryGate):
        # attempt decomposition
        operation._define()
    return control(operation, num_ctrl_qubits=num_ctrl_qubits, label=label)


def control(operation, num_ctrl_qubits=1, label=None):
    """Return controlled version of gate using controlled rotations

    Args:
        operation (Gate or Controlledgate): gate to create ControlledGate from
        num_ctrl_qubits (int): number of controls to add to gate (default=1)
        label (str): optional gate label
    Returns:
        ControlledGate: controlled version of gate. This default algorithm
            uses num_ctrl_qubits-1 ancillae qubits so returns a gate of size
            num_qubits + 2*num_ctrl_qubits - 1.

    Raises:
        QiskitError: gate contains non-gate in definitionl
    """
    from math import pi
    # pylint: disable=cyclic-import
    import qiskit.circuit.controlledgate as controlledgate
    from qiskit.circuit.quantumregister import QuantumRegister
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    # pylint: disable=unused-import
    import qiskit.extensions.standard.multi_control_rotation_gates
    import qiskit.extensions.standard.multi_control_toffoli_gate
    import qiskit.extensions.standard.multi_control_u1_gate

    q_control = QuantumRegister(num_ctrl_qubits, name='control')
    q_target = QuantumRegister(operation.num_qubits, name='target')
    q_ancillae = None  # TODO: add

    qc = QuantumCircuit(q_control, q_target)

    if operation.name == 'x' or (
            isinstance(operation, controlledgate.ControlledGate) and
            operation.base_gate.name == 'x'):
        qc.mct(q_control[:] + q_target[:-1],
               q_target[-1],
               None,
               mode='noancilla')
    elif operation.name == 'rx':
        qc.mcrx(operation.definition[0][0].params[0], q_control, q_target[0],
                use_basis_gates=True)
    elif operation.name == 'ry':
        qc.mcry(operation.definition[0][0].params[0], q_control, q_target[0],
                q_ancillae, use_basis_gates=True)
    elif operation.name == 'rz':
        qc.mcrz(operation.definition[0][0].params[0], q_control, q_target[0],
                use_basis_gates=True)
    else:
        bgate = _unroll_gate(operation, ['u1', 'u3', 'cx'])
        # now we have a bunch of single qubit rotation gates and cx
        for rule in bgate.definition:
            if rule[0].name == 'u3':
                theta, phi, lamb = rule[0].params
                if phi == -pi/2 and lamb == pi/2:
                    qc.mcrx(theta, q_control, q_target[rule[1][0].index],
                            use_basis_gates=True)
                elif phi == 0 and lamb == 0:
                    qc.mcry(theta, q_control, q_target[rule[1][0].index],
                            q_ancillae, mode='noancilla', use_basis_gates=True)
                elif theta == 0 and phi == 0:
                    qc.mcrz(lamb, q_control, q_target[rule[1][0].index],
                            use_basis_gates=True)
                else:
                    qc.mcrz(lamb, q_control, q_target[rule[1][0].index],
                            use_basis_gates=True)
                    qc.mcry(theta, q_control, q_target[rule[1][0].index],
                            q_ancillae, use_basis_gates=True)
                    qc.mcrz(phi, q_control, q_target[rule[1][0].index],
                            use_basis_gates=True)
            elif rule[0].name == 'u1':
                qc.mcu1(rule[0].params[0], q_control, q_target[rule[1][0].index])
            elif rule[0].name == 'cx':
                qc.mct(q_control[:] + [q_target[rule[1][0].index]],
                       q_target[rule[1][1].index],
                       None,
                       mode='noancilla')
            else:
                raise QiskitError('gate contains non-controllable instructions')
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
                                          definition=instr.definition)
    cgate.base_gate = base_gate
    return cgate


def _gate_to_circuit(operation):
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.circuit.quantumregister import QuantumRegister
    qr = QuantumRegister(operation.num_qubits)
    qc = QuantumCircuit(qr, name=operation.name)
    if hasattr(operation, 'definition') and operation.definition:
        for rule in operation.definition:
            if rule[0].name in {'id', 'barrier', 'measure', 'snapshot'}:
                raise QiskitError('Cannot make controlled gate with {} instruction'.format(
                    rule[0].name))
            qc.append(rule[0], qargs=[qr[bit.index] for bit in rule[1]], cargs=[])
    else:
        qc.append(operation, qargs=qr, cargs=[])
    return qc


def _unroll_gate(operation, basis_gates):
    from qiskit.converters.circuit_to_dag import circuit_to_dag
    from qiskit.converters.dag_to_circuit import dag_to_circuit
    from qiskit.transpiler.passes import Unroller
    unroller = Unroller(basis_gates)
    dag = circuit_to_dag(_gate_to_circuit(operation))
    qc = dag_to_circuit(unroller.run(dag))
    return qc.to_gate()
