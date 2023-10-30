# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,no-member

"""Benchmark utility functions."""

import numpy as np

from qiskit.quantum_info.random import random_unitary
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Reset
from qiskit.circuit.library import (
    IGate,
    U1Gate,
    U2Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    RXGate,
    RYGate,
    RZGate,
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    CRZGate,
    CU1Gate,
    CU3Gate,
    SwapGate,
    RZZGate,
    CCXGate,
    CSwapGate,
)


def random_circuit(
    n_qubits, depth, max_operands=3, measure=False, conditional=False, reset=False, seed=None
):
    """Generate random circuit of arbitrary size and form.

    Args:
        n_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum operands of each gate (between 1 and 3)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and
            conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        Exception: when invalid options given
    """
    if max_operands < 1 or max_operands > 3:
        raise Exception("max_operands must be between 1 and 3")

    one_q_ops = [
        IGate,
        U1Gate,
        U2Gate,
        U3Gate,
        XGate,
        YGate,
        ZGate,
        HGate,
        SGate,
        SdgGate,
        TGate,
        TdgGate,
        RXGate,
        RYGate,
        RZGate,
    ]
    one_param = [U1Gate, RXGate, RYGate, RZGate, RZZGate, CU1Gate, CRZGate]
    two_param = [U2Gate]
    three_param = [U3Gate, CU3Gate]
    two_q_ops = [CXGate, CYGate, CZGate, CHGate, CRZGate, CYGate, CU3Gate, SwapGate, RZZGate]
    three_q_ops = [CCXGate, CSwapGate]

    qr = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(n_qubits)

    if measure or conditional:
        cr = ClassicalRegister(n_qubits, "c")
        qc.add_register(cr)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(n_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(three_q_ops)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                value = rng.randint(0, np.power(2, n_qubits))
                op.condition = (cr, value)

            qc.append(op, register_operands)

    if measure:
        qc.measure(qr, cr)

    return qc


def build_qv_model_circuit(width, depth, seed=None):
    """
    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.
    """
    np.random.seed(seed)
    circuit = QuantumCircuit(width)
    # For each layer
    for _ in range(depth):
        # Generate uniformly random permutation Pj of [0...n-1]
        perm = np.random.permutation(width)
        # For each pair p in Pj, generate Haar random SU(4)
        for k in range(int(np.floor(width / 2))):
            U = random_unitary(4)
            pair = int(perm[2 * k]), int(perm[2 * k + 1])
            circuit.append(U, [pair[0], pair[1]])
    return circuit


def build_ripple_adder_circuit(size):
    """
    Builds a ripple adder of a given size.
    """
    n = size
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    cin = QuantumRegister(1, "cin")
    cout = QuantumRegister(1, "cout")
    ans = ClassicalRegister(n + 1, "ans")
    qc = QuantumCircuit(a, b, cin, cout, ans, name="rippleadd")

    def majority(p, a, b, c):
        """Majority gate."""
        p.cx(c, b)
        p.cx(c, a)
        p.ccx(a, b, c)

    def unmajority(p, a, b, c):
        """Unmajoritygate."""
        p.ccx(a, b, c)
        p.cx(c, a)
        p.cx(a, b)

    # Build a temporary subcircuitthat adds a to b,
    # storing the result in b
    adder_subcircuit = QuantumCircuit(cin, a, b, cout)
    majority(adder_subcircuit, cin[0], b[0], a[0])
    for j in range(n - 1):
        majority(adder_subcircuit, a[j], b[j + 1], a[j + 1])

    adder_subcircuit.cx(a[n - 1], cout[0])

    for j in reversed(range(n - 1)):
        unmajority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
        unmajority(adder_subcircuit, cin[0], b[0], a[0])

    # Set the inputs to the adder
    qc.x(a[0])  # Set input a = 0...0001
    qc.x(b)  # Set input b = 1...1111
    # Apply the adder
    qc += adder_subcircuit

    # Measure the output register in the computational basis
    for j in range(n):
        qc.measure(b[j], ans[j])
    qc.measure(cout[0], ans[n])

    return qc
