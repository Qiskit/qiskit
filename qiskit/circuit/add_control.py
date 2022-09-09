# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Add control to operation if supported."""

from typing import Union, Optional

from qiskit.circuit.exceptions import CircuitError
from qiskit.extensions import UnitaryGate
from . import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates import (MCXRecursive, RZGate, RYGate, 
                                                   CXGate, CCXGate, C3XGate, C4XGate, MCXGate)
from qiskit.quantum_info import OneQubitEulerDecomposer


def add_control(
    operation: Union[Gate, ControlledGate],
    num_ctrl_qubits: int,
    label: Union[str, None],
    ctrl_state: Union[int, str, None],
) -> ControlledGate:
    """For standard gates, if the controlled version already exists in the
    library, it will be returned (e.g. XGate.control() = CnotGate().

    For more generic gates, this method implements the controlled
    version by first decomposing into the ['u1', 'u3', 'cx'] basis, then
    controlling each gate in the decomposition.

    Open controls are implemented by conjugating the control line with
    X gates. Adds num_ctrl_qubits controls to operation.

    This function is meant to be called from the
    :method:`qiskit.circuit.gate.Gate.control()` method.

    Args:
        operation: The operation to be controlled.
        num_ctrl_qubits: The number of controls to add to gate.
        label: An optional gate label.
        ctrl_state: The control state in decimal or as a bitstring
            (e.g. '111'). If specified as a bitstring the length
            must equal num_ctrl_qubits, MSB on left. If None, use
            2**num_ctrl_qubits-1.

    Returns:
        Controlled version of gate.

    """
    if ctrl_state is None:
        ctrl_state = 2**num_ctrl_qubits - 1
    if isinstance(operation, UnitaryGate):
        # attempt decomposition
        operation._define()
    cgate = control(operation, num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
    cgate.base_gate.label = operation.label
    return cgate


def control(
    operation: Union[Gate, ControlledGate],
    num_ctrl_qubits: Optional[int] = 1,
    label: Optional[Union[None, str]] = None,
    ctrl_state: Optional[Union[None, int, str]] = None,
) -> ControlledGate:
    """Return controlled version of gate using controlled rotations. This function
    first checks the name of the operation to see if it knows of a method from which
    to generate a controlled version. Currently these are `x`, `rx`, `ry`, and `rz`.
    If a method is not directly known, it calls the unroller to convert to `u1`, `u3`,
    and `cx` gates.

    Args:
        operation: The gate used to create the ControlledGate.
        num_ctrl_qubits: The number of controls to add to gate (default=1).
        label: An optional gate label.
        ctrl_state: The control state in decimal or as
            a bitstring (e.g. '111'). If specified as a bitstring the length
            must equal num_ctrl_qubits, MSB on left. If None, use
            2**num_ctrl_qubits-1.

    Returns:
        Controlled version of gate.

    Raises:
        CircuitError: gate contains non-gate in definition
    """
    from math import pi

    # pylint: disable=cyclic-import
    from qiskit.circuit import controlledgate

    q_control = QuantumRegister(num_ctrl_qubits, name="control")
    q_target = QuantumRegister(operation.num_qubits, name="target")
    q_ancillae = None  # TODO: add
    controlled_circ = QuantumCircuit(q_control, q_target, name=f"c_{operation.name}")
    if isinstance(operation, controlledgate.ControlledGate):
        original_ctrl_state = operation.ctrl_state
    global_phase = 0
    if operation.name == "x" or (
        isinstance(operation, controlledgate.ControlledGate) and operation.base_gate.name == "x"
    ):
        controlled_circ.mct(q_control[:] + q_target[:-1], q_target[-1], q_ancillae)
        if operation.definition is not None and operation.definition.global_phase:
            global_phase += operation.definition.global_phase
    else:
        basis = ["p", "u", "x", "z", "rx", "ry", "rz", "cx"]
        if isinstance(operation, controlledgate.ControlledGate):
            operation.ctrl_state = None
        unrolled_gate = _unroll_gate(operation, basis_gates=basis)
        if unrolled_gate.definition.global_phase:
            global_phase += unrolled_gate.definition.global_phase

        definition = unrolled_gate.definition
        bit_indices = {
            bit: index
            for bits in [definition.qubits, definition.clbits]
            for index, bit in enumerate(bits)
        }

        for instruction in definition.data:
            gate, qargs = instruction.operation, instruction.qubits
            if gate.name == "x":
                controlled_circ.mct(q_control, q_target[bit_indices[qargs[0]]], q_ancillae)
            elif gate.name == "rx":
                controlled_circ.mcrx(
                    gate.definition.data[0].operation.params[0],
                    q_control,
                    q_target[bit_indices[qargs[0]]],
                    use_basis_gates=True,
                )
            elif gate.name == "ry":
                controlled_circ.mcry(
                    gate.definition.data[0].operation.params[0],
                    q_control,
                    q_target[bit_indices[qargs[0]]],
                    q_ancillae,
                    mode="noancilla",
                    use_basis_gates=True,
                )
            elif gate.name == "rz":
                controlled_circ.mcrz(
                    gate.definition.data[0].operation.params[0],
                    q_control,
                    q_target[bit_indices[qargs[0]]],
                    use_basis_gates=True,
                )
            elif gate.name == "p":
                from qiskit.circuit.library import MCPhaseGate

                controlled_circ.append(
                    MCPhaseGate(gate.params[0], num_ctrl_qubits),
                    q_control[:] + [q_target[bit_indices[qargs[0]]]],
                )
            elif gate.name == "cx":
                controlled_circ.mct(
                    q_control[:] + [q_target[bit_indices[qargs[0]]]],
                    q_target[bit_indices[qargs[1]]],
                    q_ancillae,
                )
            elif gate.name == "u":
                theta, phi, lamb = gate.params
                if num_ctrl_qubits == 1:
                    if theta == 0 and phi == 0:
                        controlled_circ.cp(lamb, q_control[0], q_target[bit_indices[qargs[0]]])
                    else:
                        controlled_circ.cu(
                            theta, phi, lamb, 0, q_control[0], q_target[bit_indices[qargs[0]]]
                        )
                else:
                    if phi == -pi / 2 and lamb == pi / 2:
                        controlled_circ.mcrx(
                            theta, q_control, q_target[bit_indices[qargs[0]]], use_basis_gates=True
                        )
                    elif phi == 0 and lamb == 0:
                        controlled_circ.mcry(
                            theta,
                            q_control,
                            q_target[bit_indices[qargs[0]]],
                            q_ancillae,
                            use_basis_gates=True,
                        )
                    elif theta == 0 and phi == 0:
                        controlled_circ.mcrz(
                            lamb, q_control, q_target[bit_indices[qargs[0]]], use_basis_gates=True
                        )
                    else:
                        mcx_control, abc_control = define_mcx_control_and_ancilla(q_control)

                        mcxr_rule = define_mcx_rule(mcx_control, q_target[bit_indices[qargs[0]]], abc_control)

                        # Getting euler angles from zyz decomposition
                        th, ph, lb, alpha = OneQubitEulerDecomposer._params_zyz(gate.to_matrix())

                        if alpha: 
                            global_phase = alpha
                            theta = th
                            phi = ph
                            lamb = lb

                        a, b, c = get_abc_matrices(phi, theta, lamb)

                        controlled_circ.unitary(c, q_target[bit_indices[qargs[0]]])
                        if abc_control is not None:
                            controlled_circ.control(abc_control)

                        controlled_circ.data += mcxr_rule

                        controlled_circ.unitary(b, q_target[bit_indices[qargs[0]]])
                        if abc_control is not None:
                            controlled_circ.control(abc_control)

                        controlled_circ.data += mcxr_rule

                        controlled_circ.unitary(a, q_target[bit_indices[qargs[0]]])
                        if abc_control is not None:
                            controlled_circ.control(abc_control)

            elif gate.name == "z":
                controlled_circ.h(q_target[bit_indices[qargs[0]]])
                controlled_circ.mcx(q_control, q_target[bit_indices[qargs[0]]], q_ancillae)
                controlled_circ.h(q_target[bit_indices[qargs[0]]])
            else:
                raise CircuitError(f"gate contains non-controllable instructions: {gate.name}")
            if gate.definition is not None and gate.definition.global_phase:
                global_phase += gate.definition.global_phase
    # apply controlled global phase
    if global_phase:
        if len(q_control) < 2:
            controlled_circ.p(global_phase, q_control)
        else:
            controlled_circ.mcp(global_phase, q_control[:-1], q_control[-1])
    if isinstance(operation, controlledgate.ControlledGate):
        operation.ctrl_state = original_ctrl_state
        new_num_ctrl_qubits = num_ctrl_qubits + operation.num_ctrl_qubits
        new_ctrl_state = operation.ctrl_state << num_ctrl_qubits | ctrl_state
        base_name = operation.base_gate.name
        base_gate = operation.base_gate
    else:
        new_num_ctrl_qubits = num_ctrl_qubits
        new_ctrl_state = ctrl_state
        base_name = operation.name
        base_gate = operation
    # In order to maintain some backward compatibility with gate names this
    # uses a naming convention where if the number of controls is <=2 the gate
    # is named like "cc<base_gate.name>", else it is named like
    # "c<num_ctrl_qubits><base_name>".
    if new_num_ctrl_qubits > 2:
        ctrl_substr = f"c{new_num_ctrl_qubits:d}"
    else:
        ctrl_substr = ("{0}" * new_num_ctrl_qubits).format("c")
    new_name = f"{ctrl_substr}{base_name}"
    cgate = controlledgate.ControlledGate(
        new_name,
        controlled_circ.num_qubits,
        operation.params,
        label=label,
        num_ctrl_qubits=new_num_ctrl_qubits,
        definition=controlled_circ,
        ctrl_state=new_ctrl_state,
        base_gate=base_gate,
    )
    return cgate

def define_mcx_rule(q_controls, q_target, q_ancilla = None):
    rule = []

    controls_and_target = q_controls + [q_target]
    if len(q_controls) == 1:
        rule += [(CXGate(), controls_and_target, [])]
    elif len(q_controls) == 2:
        rule += [(CCXGate(), controls_and_target, [])]
    elif len(q_controls) == 3:
        rule += [(C3XGate(), controls_and_target, [])]
    elif len(q_controls) == 4:
        rule += [(C4XGate(), controls_and_target, [])]
    elif len(q_controls) >= 5 and  len(q_controls) <7:
        rule += [(MCXGate(len(q_controls)), controls_and_target, [])]
    else: 
        rule = MCXRecursive(len(q_controls))._recurse(controls_and_target, q_ancilla)
    return rule

def define_mcx_control_and_ancilla(control_qubits): 
    if len(control_qubits) < 7:
        mcx_controls = list(control_qubits)
        abc_ancilla = None
    else: 
        mcx_controls = control_qubits[1:]
        abc_ancilla = control_qubits[0]
    return mcx_controls, abc_ancilla

def get_abc_matrices(lamb, theta, phi):
    # A
    a_rz = RZGate(lamb).to_matrix()
    a_ry = RYGate(theta / 2).to_matrix()
    a_matrix = a_rz.dot(a_ry)

    # B
    b_ry = RYGate(-theta / 2).to_matrix()
    b_rz = RZGate(-(phi + lamb) / 2).to_matrix()
    b_matrix = b_ry.dot(b_rz)

    # C
    c_matrix = RZGate((phi - lamb) / 2).to_matrix()

    return a_matrix, b_matrix, c_matrix

def _gate_to_dag(operation):
    from qiskit.converters.circuit_to_dag import circuit_to_dag

    if hasattr(operation, "definition") and operation.definition is not None:
        return circuit_to_dag(operation.definition)
    else:
        qr = QuantumRegister(operation.num_qubits)
        qc = QuantumCircuit(qr, name=operation.name)
        qc.append(operation, qr)
        return circuit_to_dag(qc)


def _unroll_gate(operation, basis_gates):
    from qiskit.converters.dag_to_circuit import dag_to_circuit
    from qiskit.transpiler.passes import Unroller

    unroller = Unroller(basis_gates)
    dag = _gate_to_dag(operation)
    opqc = dag_to_circuit(unroller.run(dag))
    return opqc.to_gate()
