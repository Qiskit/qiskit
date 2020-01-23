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
from . import ControlledGate, Gate, QuantumRegister, QuantumCircuit


def add_control(operation: Union[Gate, ControlledGate],
                num_ctrl_qubits: int,
                label: Union[str, None],
                ctrl_state: Union[int, str, None]) -> ControlledGate:
    """Adds num_ctrl_qubits controls to operation.

    Args:
        operation: Operation for which control will be added.
        num_ctrl_qubits: The number of controls to add to gate (default=1).
        label: Optional gate label.
            ctrl_state (int or str or None): The control state in decimal or as
                a bitstring (e.g. '111'). If None, use 2**num_ctrl_qubits-1.

    Returns:
        Controlled version of gate.

    """
    if isinstance(operation, UnitaryGate):
        # attempt decomposition
        operation._define()
    if _control_definition_known(operation, num_ctrl_qubits) and ctrl_state is None:
        return _control_predefined(operation, num_ctrl_qubits)
    return control(operation, num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)


def _control_definition_known(operation, num_ctrl_qubits):
    if num_ctrl_qubits == 2 and operation.name == 'x':
        return True
    elif num_ctrl_qubits == 1:
        return operation.name in {'x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 'swap', 'u1', 'u3', 'cx'}
    elif operation.name == 'rz' and num_ctrl_qubits > 1:
        return True
    else:
        return False


def _control_predefined(operation, num_ctrl_qubits):
    """Returns controlled gates with hard-coded definitions in
    the standard extensions."""
    if operation.name == 'x' and num_ctrl_qubits in [1, 2]:
        if num_ctrl_qubits == 1:
            import qiskit.extensions.standard.cx
            cgate = qiskit.extensions.standard.cx.CnotGate()
        else:
            import qiskit.extensions.standard.ccx
            cgate = qiskit.extensions.standard.ccx.ToffoliGate()
    elif operation.name == 'y':
        import qiskit.extensions.standard.cy
        cgate = qiskit.extensions.standard.cy.CyGate()
    elif operation.name == 'z':
        import qiskit.extensions.standard.cz
        cgate = qiskit.extensions.standard.cz.CzGate()
    elif operation.name == 'h':
        import qiskit.extensions.standard.ch
        cgate = qiskit.extensions.standard.ch.CHGate()
    elif operation.name in {'rx', 'ry', 'rz'}:
        if operation.name == 'rx':
            import qiskit.extensions.standard.crx
            cgate = qiskit.extensions.standard.crx.CrxGate(*operation.params)
        elif operation.name == 'ry':
            import qiskit.extensions.standard.cry
            cgate = qiskit.extensions.standard.cry.CryGate(*operation.params)
        else:  # operation.name == 'rz'
            import qiskit.extensions.standard.crz
            cgate = qiskit.extensions.standard.crz.CrzGate(*operation.params)
        if num_ctrl_qubits == 1:
            return cgate
        else:
            # only predefined for one control qubit
            return cgate.control(num_ctrl_qubits - 1)
    elif operation.name == 'swap':
        import qiskit.extensions.standard.cswap
        cgate = qiskit.extensions.standard.cswap.FredkinGate()
    elif operation.name == 'u1':
        import qiskit.extensions.standard.cu1
        cgate = qiskit.extensions.standard.cu1.Cu1Gate(*operation.params)
    elif operation.name == 'u3':
        import qiskit.extensions.standard.cu3
        cgate = qiskit.extensions.standard.cu3.Cu3Gate(*operation.params)
    elif operation.name == 'cx':
        import qiskit.extensions.standard.ccx
        cgate = qiskit.extensions.standard.ccx.ToffoliGate()
    else:
        raise CircuitError('No standard controlled gate for "{}"'.format(
            operation.name))
    return cgate


def control(operation: Union[Gate, ControlledGate],
            num_ctrl_qubits: Optional[int] = 1,
            label: Optional[Union[None, str]] = None,
            ctrl_state: Optional[Union[None, int, str]] = None) -> ControlledGate:
    """Return controlled version of gate using controlled rotations

    Args:
        operation: gate to create ControlledGate from
        num_ctrl_qubits: number of controls to add to gate (default=1)
        label: optional gate label
            ctrl_state (int or str or None): The control state in decimal or as
                a bitstring (e.g. '111'). If None, use 2**num_ctrl_qubits-1.

    Returns:
        Controlled version of gate.

    Raises:
        CircuitError: gate contains non-gate in definition
    """
    from math import pi
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

    if ctrl_state is not None:
        if isinstance(ctrl_state, str):
            try:
                assert(len(ctrl_state) == num_ctrl_qubits)
                ctrl_state = int(ctrl_state, 2)
            except ValueError:
                raise CircuitError(f'invalid control bit string: '
                                   '"{ctrl_state}"')
            except AssertionError:
                raise CircuitError('invalid control bit string: length != '
                                   'num_ctrl_qubits')
        _toggle_ctrl_state(qc, num_ctrl_qubits, ctrl_state)
    if operation.name == 'x' or (
            isinstance(operation, controlledgate.ControlledGate) and
            operation.base_gate_name == 'x'):
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
                raise CircuitError('gate contains non-controllable instructions')
    if ctrl_state is not None:
        _toggle_ctrl_state(qc, num_ctrl_qubits, ctrl_state)
    instr = qc.to_instruction()
    if isinstance(operation, controlledgate.ControlledGate):
        new_num_ctrl_qubits = num_ctrl_qubits + operation.num_ctrl_qubits
        base_name = operation.base_gate_name
        base_gate = operation.base_gate
        base_gate_name = operation.base_gate_name
    else:
        new_num_ctrl_qubits = num_ctrl_qubits
        base_name = operation.name
        base_gate = operation.__class__
        base_gate_name = operation.name
    # In order to maintain some backward compatibility with gate names this
    # uses a naming convention where if the number of controls is <=2 the gate
    # is named like "cc<base_gate_name>", else it is named like
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
    cgate.base_gate_name = base_gate_name
    return cgate


def _toggle_ctrl_state(qc: QuantumCircuit,
                       num_qubits: int,
                       ctrl_state: int) -> Gate:
    if isinstance(ctrl_state, int) and 0 <= ctrl_state < 2**num_qubits:
        bit_ctrl_state = bin(ctrl_state)[2:].zfill(num_qubits)
    else:
        raise CircuitError('invalid control state specified')
    for ind, val in enumerate(bit_ctrl_state):
        if val == '0':
            qc.x(num_qubits - ind - 1)


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


def _unroll_gate(operation, basis_gates):
    from qiskit.converters.circuit_to_dag import circuit_to_dag
    from qiskit.converters.dag_to_circuit import dag_to_circuit
    from qiskit.transpiler.passes import Unroller
    unroller = Unroller(basis_gates)
    dag = circuit_to_dag(_gate_to_circuit(operation))
    qc = dag_to_circuit(unroller.run(dag))
    return qc.to_gate()
