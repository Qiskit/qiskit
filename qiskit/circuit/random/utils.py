# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for generating random circuits."""

from inspect import signature
import numpy as np

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Reset
from qiskit.extensions import (IdGate, U1Gate, U2Gate, U3Gate, XGate,
                               YGate, ZGate, HGate, SGate, SdgGate, TGate,
                               TdgGate, RXGate, RYGate, RZGate, CnotGate,
                               CyGate, CzGate, CHGate, CrzGate, Cu1Gate,
                               Cu3Gate, SwapGate, RZZGate,
                               ToffoliGate, FredkinGate)
from qiskit.exceptions import QiskitError


def random_circuit(n_qubits, depth, max_operands=3, measure=False,
                   conditional=False, reset=False, seed=None):
    """Generate random circuit of arbitrary size and form.

    Args:
        n_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum operands of each gate (between 1 and 3)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        QiskitError: when invalid options given
    """
    if max_operands < 1 or max_operands > 3:
        raise QiskitError("max_operands must be between 1 and 3")

    one_q_ops = [IdGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate,
                 HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate]
    two_q_ops = [CnotGate, CyGate, CzGate, CHGate, CrzGate,
                 Cu1Gate, Cu3Gate, SwapGate, RZZGate]
    three_q_ops = [ToffoliGate, FredkinGate]

    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(n_qubits)

    if measure or conditional:
        cr = ClassicalRegister(n_qubits, 'c')
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
            op_args = list(signature(operation).parameters.keys())
            num_angles = len(op_args)
            if 'label' in op_args:  # TODO: label is not part of gate params and must be removed
                num_angles -= 1
            angles = [rng.uniform(0, 2*np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                possible_values = range(pow(2, n_qubits))
                value = rng.choice(list(possible_values))
                op.control = (cr, value)

            qc.append(op, register_operands)

    if measure:
        qc.measure(qr, cr)

    return qc
