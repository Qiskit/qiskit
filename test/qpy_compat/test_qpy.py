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

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.classicalregister import Clbit
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.qpy_serialization import dump, load
from qiskit.opflow import X, Y, Z
from qiskit.quantum_info.random import random_unitary
from qiskit.circuit.library import U1Gate, U2Gate, U3Gate


def generate_circuits():
    """Generate reference circuits."""
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

    unitary_circuit = QuantumCircuit(5)
    unitary_circuit.unitary(random_unitary(32, seed=100), [0, 1, 2, 3, 4])
    unitary_circuit.measure_all()

    random_circuits = []
    for i in range(10):
        random_circuits.append(
            random_circuit(10, 10, measure=True, conditional=True, reset=True, seed=42 + i)
        )

    string_parameters = (X ^ Y ^ Z).to_circuit_op().to_circuit()

    register_edge_cases = []
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

    param_circuit = QuantumCircuit(1)
    theta = Parameter("theta")
    lam = Parameter("λ")
    theta_pi = 3.14159 * theta
    pe = theta_pi / lam
    param_circuit.append(U3Gate(theta, theta_pi, lam), [0])
    param_circuit.append(U1Gate(pe), [0])
    param_circuit.append(U2Gate(theta_pi, lam), [0])

    return {
        "full.qpy": [full_circuit],
        "unitary.qpy": [unitary_circuit],
        "multiple.qpy": random_circuits,
        "string_parameters.qpy": [string_parameters],
        "register_edge_cases.qpy": register_edge_cases,
        "parameterized.qpy": [param_circuit],
    }


def assert_equal(reference, qpy, count, bind=None):
    """Compare two circuits."""
    if bind is not None:
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
    if not bind and reference.name != qpy.name:
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
        print("Loading qpy file: %s" % path)
        with open(path, "rb") as fd:
            qpy_circuits = load(fd)
        for i, circuit in enumerate(circuits):
            bind = None
            if path == "parameterized.qpy":
                bind = [1, 2]
            assert_equal(circuit, qpy_circuits[i], i, bind=bind)


def _main():
    parser = argparse.ArgumentParser(description="Test QPY backwards compatibilty")
    parser.add_argument("command", choices=["generate", "load"])
    args = parser.parse_args()
    qpy_files = generate_circuits()
    if args.command == "generate":
        generate_qpy(qpy_files)
    else:
        load_qpy(qpy_files)


if __name__ == "__main__":
    _main()
