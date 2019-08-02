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
import numpy as np
from qiskit import QiskitError


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
    if control_definition_known(operation, num_ctrl_qubits):
        return q_if_predefined(operation, num_ctrl_qubits)
    mode = 'qi'
    if mode == 'naive':
        q_if_fn = q_if_naive
    elif mode == 'qi':
        q_if_fn = q_if_qi
    elif mode == 'aqua':
        q_if_fn = q_if_aqua
    return q_if_fn(operation, num_ctrl_qubits=num_ctrl_qubits, label=label)


def control_definition_known(operation, num_ctrl_qubits):
    if num_ctrl_qubits == 2 and operation.name == 'x':
        return True
    elif num_ctrl_qubits == 1:
        return operation.name in {'x', 'y', 'z', 'h', 'rz', 'swap', 'u1', 'u3', 'cx'}


def q_if_predefined(operation, num_ctrl_qubits):
    if operation.name == 'x':
        import qiskit.extensions.standard.cx
        return qiskit.extensions.standard.cx.CnotGate()
    elif operation.name == 'y':
        import qiskit.extensions.standard.cy
        return qiskit.extensions.standard.cy.CyGate()
    elif operation.name == 'z':
        import qiskit.extensions.standard.cz
        return qiskit.extensions.standard.cz.CzGate()
    elif operation.name == 'h':
        import qiskit.extensions.standard.ch
        return qiskit.extensions.standard.ch.CHGate()
    elif operation.name == 'rz':
        import qiskit.extensions.standard.crz
        return qiskit.extensions.standard.crz.CrzGate(*operation.params)
    elif operation.name == 'swap':
        import qiskit.extensions.standard.cswap
        return qiskit.extensions.standard.cswap.FredkinGate()
    elif operation.name == 'u1':
        import qiskit.extensions.standard.cu1
        return qiskit.extensions.standard.cu1.Cu1Gate(*operation.params)
    elif operation.name == 'u3':
        import qiskit.extensions.standard.cu3
        return qiskit.extensions.standard.cu3.Cu3Gate(*operation.params)
    elif operation.name == 'cx':
        import qiskit.extensions.standard.ccx
        return qiskit.extensions.standard.ccx.ToffoliGate()
    else:
        raise QiskitError('No standard controlled gate for "{}"'.format(
            operation.name))


def q_if_naive(operation, num_ctrl_qubits=1, label=None):
    """Return controlled version of gate

    Args:
        num_ctrl_qubits (int): number of controls to add to gate (default=1)
        label (str): optional gate label

    Returns:
        ControlledGate: controlled version of gate. This default algorithm
            uses num_ctrl_qubits-1 ancillae qubits so returns a gate of size
            num_qubits + 2*num_ctrl_qubits - 1.

    Raises:
        AttributeError: unrecognized gate from standard extensions
        QiskitError: gate definition contains non-gate
    """
    basis_gates = {'u1', 'u3', 'id', 'cx'}
    # pylint: disable=cyclic-import
    import qiskit.circuit.controlledgate as controlledgate
    from qiskit.circuit.quantumregister import QuantumRegister
    cgate = None
    if not (hasattr(operation, 'definition') and operation.definition is not None):
        if operation.name in ['u3', 'cx']:
            cgate = operation.q_if()
            return cgate.q_if(num_ctrl_qubits=num_ctrl_qubits-1)
        else:
            raise AttributeError('unexpected QASM base gate: {}'.format(operation.name))
    elif hasattr(operation, 'definition') and operation.definition is not None:
        definition = []
        if num_ctrl_qubits > 0:
            #  we need to add +1 control, unroll
            bgate = _unroll_gate(operation, basis_gates)
            qr = QuantumRegister(operation.num_qubits + 1)
            for rule in bgate.definition:
                if hasattr(rule[0], 'q_if'):
                    bgate_bits = list(
                        [qr[0]]
                        + [qr[1 + bit.index] for bit in rule[1]])
                    q_if_rule = (rule[0].q_if(), bgate_bits, [])
                    definition.append(q_if_rule)
                else:
                    raise QiskitError('gate contains non-controllable intruction')
            if isinstance(operation, controlledgate.ControlledGate):
                # pylint: disable=no-member
                this_num_ctrl_qubits = operation.num_ctrl_qubits
            else:
                this_num_ctrl_qubits = 1
            cgate = controlledgate.ControlledGate(
                'c{}'.format(operation.name),
                operation.num_qubits+1,
                operation.params,
                label=label,
                num_ctrl_qubits=this_num_ctrl_qubits,
                definition=definition)
            return cgate.q_if(num_ctrl_qubits=num_ctrl_qubits-1)
        else:
            # ok stop depth search
            # pylint: inconsistent-return-statements
            return operation
    return controlledgate.ControlledGate(
        'c{}'.format(operation.name),
        operation.num_qubits+1,
        operation.params,
        label=label,
        num_ctrl_qubits=this_num_ctrl_qubits)

def q_if_qi(operation, num_ctrl_qubits=1, label=None):
    """Return controlled version of gate using multiplexed rotations

    Args:
        num_ctrl_qubits (int): number of controls to add to gate (default=1)
        label (str): optional gate label
    Returns:
        ControlledGate: controlled version of gate. This default algorithm
            uses num_ctrl_qubits-1 ancillae qubits so returns a gate of size
            num_qubits + 2*num_ctrl_qubits - 1.

    Raises:
        QiskitError: gate contains non-gate in definitionl
    """
    # pylint: disable=cyclic-import
    import qiskit.circuit.controlledgate as controlledgate
    from qiskit.circuit.quantumregister import QuantumRegister
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.extensions.quantum_initializer.ucrot import UCRot
    from math import pi
    bgate = _unroll_gate(operation, ['u3', 'cx'])
    # now we have a bunch of single qubit rotation gates and cx
    q_control = QuantumRegister(num_ctrl_qubits)
    q_target = QuantumRegister(operation.num_qubits)
    qc = QuantumCircuit(q_control, q_target)
    for rule in bgate.definition:
        if rule[0].name == 'u3':
            theta, phi, lamb = [np.repeat([[0], [angle]], [2**num_ctrl_qubits - 1, 1])
                                    for angle in rule[0].params]
            qc.append(UCRot(lamb.tolist(), 'Z'), qargs=q_control[:] + [q_target[rule[1][0].index]])
            qc.append(UCRot(theta.tolist(), 'Y'), qargs=q_control[:] + [q_target[rule[1][0].index]])
            qc.append(UCRot(phi.tolist(), 'Z'), qargs=q_control[:] + [q_target[rule[1][0].index]])
        elif rule[0].name == 'cx':
            theta = np.repeat([[0], [pi]], [2**(num_ctrl_qubits+1) - 1, 1])
            qc.append(UCRot(theta.tolist(), 'X'), qargs=q_control[:] + q_target[:])
        else:
            raise QiskitError('gate contains non-controllable intructions')
    instr = qc.to_instruction()
    cgate = controlledgate.ControlledGate('c{0:d}{1}'.format(
        num_ctrl_qubits, operation.name),
                                          instr.num_qubits,
                                          instr.params,
                                          label=label,
                                          num_ctrl_qubits=num_ctrl_qubits,
                                          definition=instr.definition)
    return cgate


def q_if_aqua(operation, num_ctrl_qubits=1, label=None):
    """Return controlled version of gate using controlled rotations from aqua

    Args:
        num_ctrl_qubits (int): number of controls to add to gate (default=1)
        label (str): optional gate label
    Returns:
        ControlledGate: controlled version of gate. This default algorithm
            uses num_ctrl_qubits-1 ancillae qubits so returns a gate of size
            num_qubits + 2*num_ctrl_qubits - 1.

    Raises:
        QiskitError: gate contains non-gate in definitionl
    """
    # pylint: disable=cyclic-import
    import qiskit.circuit.controlledgate as controlledgate
    from qiskit.circuit.quantumregister import QuantumRegister
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    import qiskit.aqua
    # if isinstance(operation, controlledgate.ControlledGate):
    #     # pylint: disable=no-member
    #     new_num_ctrl_qubits = operation.num_ctrl_qubits + num_ctrl_qubits
    # else:
    #     new_num_ctrl_qubits = num_ctrl_qubits
    bgate = _unroll_gate(operation, ['u3', 'cx'])
    # now we have a bunch of single qubit rotation gates and cx
    q_control = QuantumRegister(num_ctrl_qubits)
    q_target = QuantumRegister(operation.num_qubits)
    qc = QuantumCircuit(q_control, q_target)
    for rule in bgate.definition:
        if rule[0].name == 'u3':
            theta, phi, lamb = rule[0].params
            qc.mcrz(lamb, q_control, q_target[rule[1][0].index], use_basis_gates=True)
            qc.mcry(theta, q_control, q_target[rule[1][0].index], None, mode='noancilla',
                    use_basis_gates=True)
            qc.mcrz(phi, q_control, q_target[rule[1][0].index], use_basis_gates=True)
        elif rule[0].name == 'cx':
            qc.mct(q_control[:] + [q_target[rule[1][0].index]],
                   q_target[rule[1][1].index],
                   None,
                   mode='noancilla')
        else:
            raise QiskitError('gate contains non-controllable intructions')
    instr = qc.to_instruction()
    cgate = controlledgate.ControlledGate('c{0:d}{1}'.format(
        num_ctrl_qubits, operation.name),
                                          instr.num_qubits,
                                          instr.params,
                                          label=label,
                                          num_ctrl_qubits=num_ctrl_qubits,
                                          definition=instr.definition)
    return cgate


def _gate_to_circuit(operation):
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.circuit.quantumregister import QuantumRegister
    qr = QuantumRegister(operation.num_qubits)
    qc = QuantumCircuit(qr, name=operation.name)
    if hasattr(operation, 'definition') and operation.definition:
        for rule in operation.definition:
            qc.append(rule[0], qargs=[qr[bit.index] for bit in rule[1]], cargs=[])
    else:
        qc.append(operation, qargs=qr, cargs=[])
    return qc


def _unroll_gate(operation, basis_gates):
    from qiskit.converters.circuit_to_dag import circuit_to_dag
    from qiskit.converters.dag_to_circuit import dag_to_circuit
    from qiskit.converters.instruction_to_gate import instruction_to_gate
    from qiskit.transpiler.passes import Unroller
    unroller = Unroller(basis_gates)
    dag = circuit_to_dag(_gate_to_circuit(operation))
    qc = dag_to_circuit(unroller.run(dag))
    return instruction_to_gate(qc.to_instruction())
