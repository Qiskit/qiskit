#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases to verify qpy backwards compatibility."""

import argparse
import random
import sys

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classicalregister import Clbit
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.qpy_serialization import dump, load
from qiskit.opflow import X, Y, Z, I
from qiskit.quantum_info.random import random_unitary
from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, QFT


def generate_full_circuit():
    """Generate a multiregister circuit with name, metadata, phase."""
    qr_a = QuantumRegister(4, "a")
    qr_b = QuantumRegister(4, "b")
    cr_c = ClassicalRegister(4, "c")
    cr_d = ClassicalRegister(4, "d")
    full_circuit = QuantumCircuit(
        qr_a,
        qr_b,
        cr_c,
        cr_d,
        name="MyCircuit",
        metadata={"test": 1, "a": 2},
        global_phase=3.14159,
    )
    full_circuit.h(qr_a)
    full_circuit.cx(qr_a, qr_b)
    full_circuit.barrier(qr_a)
    full_circuit.barrier(qr_b)
    full_circuit.measure(qr_a, cr_c)
    full_circuit.measure(qr_b, cr_d)
    return full_circuit


def generate_unitary_gate_circuit():
    """Generate a circuit with a unitary gate."""
    unitary_circuit = QuantumCircuit(5)
    unitary_circuit.unitary(random_unitary(32, seed=100), [0, 1, 2, 3, 4])
    unitary_circuit.measure_all()
    return unitary_circuit


def generate_random_circuits():
    """Generate multiple random circuits."""
    random_circuits = []
    for i in range(1, 15):
        qc = QuantumCircuit(i)
        qc.h(0)
        if i > 1:
            for j in range(i - 1):
                qc.cx(0, j + 1)
        qc.measure_all()
        for j in range(i):
            qc.reset(j)
        qc.x(0).c_if(qc.cregs[0], i)
        for j in range(i):
            qc.measure(j, j)
        random_circuits.append(qc)
    return random_circuits


def generate_string_parameters():
    """Generate a circuit from pauli tensor opflow."""
    return (X ^ Y ^ Z).to_circuit_op().to_circuit()


def generate_register_edge_cases():
    """Generate register edge case circuits."""
    register_edge_cases = []
    # Circuit with shared bits in a register
    qubits = [Qubit() for _ in range(5)]
    shared_qc = QuantumCircuit()
    shared_qc.add_bits(qubits)
    shared_qr = QuantumRegister(bits=qubits)
    shared_qc.add_register(shared_qr)
    shared_qc.h(shared_qr)
    shared_qc.cx(0, 1)
    shared_qc.cx(0, 2)
    shared_qc.cx(0, 3)
    shared_qc.cx(0, 4)
    shared_qc.measure_all()
    register_edge_cases.append(shared_qc)
    # Circuit with registers that have a mix of standalone and shared register
    # bits
    qr = QuantumRegister(5, "foo")
    qr = QuantumRegister(name="bar", bits=qr[:3] + [Qubit(), Qubit()])
    cr = ClassicalRegister(5, "foo")
    cr = ClassicalRegister(name="classical_bar", bits=cr[:3] + [Clbit(), Clbit()])
    hybrid_qc = QuantumCircuit(qr, cr)
    hybrid_qc.h(0)
    hybrid_qc.cx(0, 1)
    hybrid_qc.cx(0, 2)
    hybrid_qc.cx(0, 3)
    hybrid_qc.cx(0, 4)
    hybrid_qc.measure(qr, cr)
    register_edge_cases.append(hybrid_qc)
    # Circuit with mixed standalone and shared registers
    qubits = [Qubit() for _ in range(5)]
    clbits = [Clbit() for _ in range(5)]
    mixed_qc = QuantumCircuit()
    mixed_qc.add_bits(qubits)
    mixed_qc.add_bits(clbits)
    qr = QuantumRegister(bits=qubits)
    cr = ClassicalRegister(bits=clbits)
    mixed_qc.add_register(qr)
    mixed_qc.add_register(cr)
    qr_standalone = QuantumRegister(2, "standalone")
    mixed_qc.add_register(qr_standalone)
    cr_standalone = ClassicalRegister(2, "classical_standalone")
    mixed_qc.add_register(cr_standalone)
    mixed_qc.unitary(random_unitary(32, seed=42), qr)
    mixed_qc.unitary(random_unitary(4, seed=100), qr_standalone)
    mixed_qc.measure(qr, cr)
    mixed_qc.measure(qr_standalone, cr_standalone)
    register_edge_cases.append(mixed_qc)
    # Circuit with out of order register bits
    qr_standalone = QuantumRegister(2, "standalone")
    qubits = [Qubit() for _ in range(5)]
    clbits = [Clbit() for _ in range(5)]
    ooo_qc = QuantumCircuit()
    ooo_qc.add_bits(qubits)
    ooo_qc.add_bits(clbits)
    random.seed(42)
    random.shuffle(qubits)
    random.shuffle(clbits)
    qr = QuantumRegister(bits=qubits)
    cr = ClassicalRegister(bits=clbits)
    ooo_qc.add_register(qr)
    ooo_qc.add_register(cr)
    qr_standalone = QuantumRegister(2, "standalone")
    cr_standalone = ClassicalRegister(2, "classical_standalone")
    ooo_qc.add_bits([qr_standalone[1], qr_standalone[0]])
    ooo_qc.add_bits([cr_standalone[1], cr_standalone[0]])
    ooo_qc.add_register(qr_standalone)
    ooo_qc.add_register(cr_standalone)
    ooo_qc.unitary(random_unitary(32, seed=42), qr)
    ooo_qc.unitary(random_unitary(4, seed=100), qr_standalone)
    ooo_qc.measure(qr, cr)
    ooo_qc.measure(qr_standalone, cr_standalone)
    register_edge_cases.append(ooo_qc)
    return register_edge_cases


def generate_parameterized_circuit():
    """Generate a circuit with parameters and parameter expressions."""
    param_circuit = QuantumCircuit(1)
    theta = Parameter("theta")
    lam = Parameter("λ")
    theta_pi = 3.14159 * theta
    pe = theta_pi / lam
    param_circuit.append(U3Gate(theta, theta_pi, lam), [0])
    param_circuit.append(U1Gate(pe), [0])
    param_circuit.append(U2Gate(theta_pi, lam), [0])
    return param_circuit


def generate_qft_circuit():
    """Generate a QFT circuit with initialization."""
    k = 5
    state = (1 / np.sqrt(8)) * np.array(
        [
            np.exp(-1j * 2 * np.pi * k * (0) / 8),
            np.exp(-1j * 2 * np.pi * k * (1) / 8),
            np.exp(-1j * 2 * np.pi * k * (2) / 8),
            np.exp(-1j * 2 * np.pi * k * 3 / 8),
            np.exp(-1j * 2 * np.pi * k * 4 / 8),
            np.exp(-1j * 2 * np.pi * k * 5 / 8),
            np.exp(-1j * 2 * np.pi * k * 6 / 8),
            np.exp(-1j * 2 * np.pi * k * 7 / 8),
        ]
    )

    qubits = 3
    qft_circ = QuantumCircuit(qubits, qubits)
    qft_circ.initialize(state)
    qft_circ.append(QFT(qubits), range(qubits))
    qft_circ.measure(range(qubits), range(qubits))
    return qft_circ


def generate_param_phase():
    """Generate circuits with parameterize global phase."""
    output_circuits = []
    # Generate circuit with ParameterExpression global phase
    theta = Parameter("theta")
    phi = Parameter("phi")
    sum_param = theta + phi
    qc = QuantumCircuit(5, 1, global_phase=sum_param)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.barrier()
    qc.rz(sum_param, range(3))
    qc.rz(phi, 3)
    qc.rz(theta, 4)
    qc.barrier()
    for i in reversed(range(4)):
        qc.cx(i, i + 1)
    qc.h(0)
    qc.measure(0, 0)
    output_circuits.append(qc)
    # Generate circuit with Parameter global phase
    theta = Parameter("theta")
    bell_qc = QuantumCircuit(2, global_phase=theta)
    bell_qc.h(0)
    bell_qc.cx(0, 1)
    bell_qc.measure_all()
    output_circuits.append(bell_qc)
    return output_circuits


def generate_single_clbit_condition_teleportation():  # pylint: disable=invalid-name
    """Generate single clbit condition teleportation circuit."""
    qr = QuantumRegister(1)
    cr = ClassicalRegister(2, name="name")
    teleport_qc = QuantumCircuit(qr, cr, name="Reset Test")
    teleport_qc.x(0)
    teleport_qc.measure(0, cr[0])
    teleport_qc.x(0).c_if(cr[0], 1)
    teleport_qc.measure(0, cr[1])
    return teleport_qc


def generate_parameter_vector():
    """Generate tests for parameter vector element ordering."""
    qc = QuantumCircuit(11)
    input_params = ParameterVector("x_par", 11)
    user_params = ParameterVector("θ_par", 11)
    for i, param in enumerate(user_params):
        qc.ry(param, i)
    for i, param in enumerate(input_params):
        qc.rz(param, i)
    return qc


def generate_parameter_vector_expression():  # pylint: disable=invalid-name
    """Generate tests for parameter vector element ordering."""
    qc = QuantumCircuit(7)
    entanglement = [[i, i + 1] for i in range(7 - 1)]
    input_params = ParameterVector("x_par", 14)
    user_params = ParameterVector("\u03B8_par", 1)

    for i in range(qc.num_qubits):
        qc.ry(user_params[0], qc.qubits[i])

    for source, target in entanglement:
        qc.cz(qc.qubits[source], qc.qubits[target])

    for i in range(qc.num_qubits):
        qc.rz(-2 * input_params[2 * i + 1], qc.qubits[i])
        qc.rx(-2 * input_params[2 * i], qc.qubits[i])

    return qc


def generate_evolution_gate():
    """Generate a circuit with a pauli evolution gate."""
    # Runtime import since this only exists in terra 0.19.0
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import SuzukiTrotter

    synthesis = SuzukiTrotter()
    evo = PauliEvolutionGate([(Z ^ I) + (I ^ Z)] * 5, time=2.0, synthesis=synthesis)
    qc = QuantumCircuit(2)
    qc.append(evo, range(2))
    return qc


def generate_control_flow_circuits():
    """Test qpy serialization with control flow instructions."""
    from qiskit.circuit.controlflow import WhileLoopOp, IfElseOp, ForLoopOp

    # If instruction
    circuits = []
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    true_body = QuantumCircuit(1)
    true_body.x(0)
    if_op = IfElseOp((qc.clbits[0], True), true_body=true_body)
    qc.append(if_op, [1])
    qc.measure(1, 1)
    circuits.append(qc)
    # If else instruction
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    false_body = QuantumCircuit(1)
    false_body.y(0)
    if_else_op = IfElseOp((qc.clbits[0], True), true_body, false_body)
    qc.append(if_else_op, [1])
    qc.measure(1, 1)
    circuits.append(qc)
    # While loop
    qc = QuantumCircuit(2, 1)
    block = QuantumCircuit(2, 1)
    block.h(0)
    block.cx(0, 1)
    block.measure(0, 0)
    while_loop = WhileLoopOp((qc.clbits[0], 0), block)
    qc.append(while_loop, [0, 1], [0])
    circuits.append(qc)
    # for loop range
    qc = QuantumCircuit(2, 1)
    body = QuantumCircuit(2, 1)
    body.h(0)
    body.cx(0, 1)
    body.measure(0, 0)
    body.break_loop().c_if(0, True)
    for_loop_op = ForLoopOp(range(5), None, body=body)
    qc.append(for_loop_op, [0, 1], [0])
    circuits.append(qc)
    # For loop iterator
    qc = QuantumCircuit(2, 1)
    for_loop_op = ForLoopOp(iter(range(5)), None, body=body)
    qc.append(for_loop_op, [0, 1], [0])
    circuits.append(qc)
    return circuits


def generate_circuits(version_str=None):
    """Generate reference circuits."""
    version_parts = None
    if version_str:
        version_parts = tuple(int(x) for x in version_str.split("."))

    output_circuits = {
        "full.qpy": [generate_full_circuit()],
        "unitary.qpy": [generate_unitary_gate_circuit()],
        "multiple.qpy": generate_random_circuits(),
        "string_parameters.qpy": [generate_string_parameters()],
        "register_edge_cases.qpy": generate_register_edge_cases(),
        "parameterized.qpy": [generate_parameterized_circuit()],
    }
    if version_parts is None:
        return output_circuits

    if version_parts >= (0, 18, 1):
        output_circuits["qft_circuit.qpy"] = [generate_qft_circuit()]
        output_circuits["teleport.qpy"] = [generate_single_clbit_condition_teleportation()]

    if version_parts >= (0, 19, 0):
        output_circuits["param_phase.qpy"] = generate_param_phase()

    if version_parts >= (0, 19, 1):
        output_circuits["parameter_vector.qpy"] = [generate_parameter_vector()]
        output_circuits["pauli_evo.qpy"] = [generate_evolution_gate()]
        output_circuits["parameter_vector_expression.qpy"] = [
            generate_parameter_vector_expression()
        ]
    if version_parts >= (0, 19, 2):
        output_circuits["control_flow.qpy"] = generate_control_flow_circuits()

    return output_circuits


def assert_equal(reference, qpy, count, bind=None):
    """Compare two circuits."""
    if bind is not None:
        reference_parameter_names = [x.name for x in reference.parameters]
        qpy_parameter_names = [x.name for x in qpy.parameters]
        if reference_parameter_names != qpy_parameter_names:
            msg = (
                f"Circuit {count} parameter mismatch:"
                f" {reference_parameter_names} != {qpy_parameter_names}"
            )
            sys.stderr.write(msg)
            sys.exit(4)
        reference = reference.bind_parameters(bind)
        qpy = qpy.bind_parameters(bind)
    if reference != qpy:
        msg = (
            f"Reference Circuit {count}:\n{reference}\nis not equivalent to "
            f"qpy loaded circuit {count}:\n{qpy}\n"
        )
        sys.stderr.write(msg)
        sys.exit(1)
    # Don't compare name on bound circuits
    if bind is None and reference.name != qpy.name:
        msg = f"Circuit {count} name mismatch {reference.name} != {qpy.name}"
        sys.stderr.write(msg)
        sys.exit(2)
    if reference.metadata != qpy.metadata:
        msg = f"Circuit {count} metadata mismatch: {reference.metadata} != {qpy.metadata}"
        sys.stderr.write(msg)
        sys.exit(3)


def generate_qpy(qpy_files):
    """Generate qpy files from reference circuits."""
    for path, circuits in qpy_files.items():
        with open(path, "wb") as fd:
            dump(circuits, fd)


def load_qpy(qpy_files):
    """Load qpy circuits from files and compare to reference circuits."""
    for path, circuits in qpy_files.items():
        print(f"Loading qpy file: {path}")
        with open(path, "rb") as fd:
            qpy_circuits = load(fd)
        for i, circuit in enumerate(circuits):
            bind = None
            if path == "parameterized.qpy":
                bind = [1, 2]
            elif path == "param_phase.qpy":
                if i == 0:
                    bind = [1, 2]
                else:
                    bind = [1]
            elif path == "parameter_vector.qpy":
                bind = np.linspace(1.0, 2.0, 22)
            elif path == "parameter_vector_expression.qpy":
                bind = np.linspace(1.0, 2.0, 15)

            assert_equal(circuit, qpy_circuits[i], i, bind=bind)


def _main():
    parser = argparse.ArgumentParser(description="Test QPY backwards compatibilty")
    parser.add_argument("command", choices=["generate", "load"])
    parser.add_argument(
        "--version",
        "-v",
        help=(
            "Optionally specify the version being tested. "
            "This will enable additional circuit features "
            "to test generating and loading QPY."
        ),
    )
    args = parser.parse_args()
    qpy_files = generate_circuits(args.version)
    if args.command == "generate":
        generate_qpy(qpy_files)
    else:
        load_qpy(qpy_files)


if __name__ == "__main__":
    _main()
