# -*- coding: utf-8 -*-

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

"""Example of using the StochasticSwap pass."""

from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler import CouplingMap, Layout
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])
qr = QuantumRegister(2, 'q')
ar = QuantumRegister(2, 'a')
cr = ClassicalRegister(4, 'c')
circ = QuantumCircuit(qr, ar, cr)
circ.cx(qr[1], ar[0])
circ.cx(qr[0], ar[1])
circ.measure(qr[0], cr[0])
circ.h(qr)
circ.h(ar)
circ.cx(qr[0], qr[1])
circ.cx(ar[0], ar[1])
circ.measure(qr[0], cr[0])
circ.measure(qr[1], cr[1])
circ.measure(ar[0], cr[2])
circ.measure(ar[1], cr[3])
dag = circuit_to_dag(circ)
#                                             ┌─┐┌───┐        ┌─┐
# q_0: |0>─────────────────■──────────────────┤M├┤ H ├──■─────┤M├
#                   ┌───┐  │                  └╥┘└───┘┌─┴─┐┌─┐└╥┘
# q_1: |0>──■───────┤ H ├──┼───────────────────╫──────┤ X ├┤M├─╫─
#         ┌─┴─┐┌───┐└───┘  │               ┌─┐ ║      └───┘└╥┘ ║
# a_0: |0>┤ X ├┤ H ├───────┼─────────■─────┤M├─╫────────────╫──╫─
#         └───┘└───┘     ┌─┴─┐┌───┐┌─┴─┐┌─┐└╥┘ ║            ║  ║
# a_1: |0>───────────────┤ X ├┤ H ├┤ X ├┤M├─╫──╫────────────╫──╫─
#                        └───┘└───┘└───┘└╥┘ ║  ║            ║  ║
#  c_0: 0 ═══════════════════════════════╬══╬══╩════════════╬══╩═
#                                        ║  ║               ║
#  c_1: 0 ═══════════════════════════════╬══╬═══════════════╩════
#                                        ║  ║
#  c_2: 0 ═══════════════════════════════╬══╩════════════════════
#                                        ║
#  c_3: 0 ═══════════════════════════════╩═══════════════════════
#
#                                ┌─┐┌───┐                     ┌─┐
# q_0: |0>────────────────────■──┤M├┤ H ├──────────────────■──┤M├──────
#                           ┌─┴─┐└╥┘└───┘┌───┐┌───┐      ┌─┴─┐└╥┘┌─┐
# q_1: |0>──■───X───────────┤ X ├─╫──────┤ H ├┤ X ├─X────┤ X ├─╫─┤M├───
#         ┌─┴─┐ │      ┌───┐└───┘ ║      └───┘└─┬─┘ │    └───┘ ║ └╥┘┌─┐
# a_0: |0>┤ X ├─┼──────┤ H ├──────╫─────────────■───┼──────────╫──╫─┤M├
#         └───┘ │ ┌───┐└───┘      ║                 │ ┌─┐      ║  ║ └╥┘
# a_1: |0>──────X─┤ H ├───────────╫─────────────────X─┤M├──────╫──╫──╫─
#                 └───┘           ║                   └╥┘      ║  ║  ║
#  c_0: 0 ════════════════════════╩════════════════════╬═══════╩══╬══╬═
#                                                      ║          ║  ║
#  c_1: 0 ═════════════════════════════════════════════╬══════════╩══╬═
#                                                      ║             ║
#  c_2: 0 ═════════════════════════════════════════════╬═════════════╩═
#                                                      ║
#  c_3: 0 ═════════════════════════════════════════════╩═══════════════
#
# Layout from mapper:
# {qr[0]: 0,
#  qr[1]: 1,
#  ar[0]: 2,
#  ar[1]: 3}
#
#     2
#     |
# 0 - 1 - 3
# Build the expected output to verify the pass worked
expected = QuantumCircuit(qr, ar, cr)
expected.cx(qr[1], ar[0])
expected.swap(qr[0], qr[1])
expected.cx(qr[1], ar[1])
expected.h(ar[1])
expected.h(ar[0])
expected.measure(qr[1], cr[0])
expected.h(qr[0])
expected.swap(qr[1], ar[1])
expected.h(ar[1])
expected.cx(ar[0], qr[1])
expected.measure(ar[0], cr[2])
expected.swap(qr[1], ar[1])
expected.measure(ar[1], cr[3])
expected.cx(qr[1], qr[0])
expected.measure(qr[1], cr[0])
expected.measure(qr[0], cr[1])
expected_dag = circuit_to_dag(expected)

layout = Layout({qr[0]: 0, qr[1]: 1, ar[0]: 2, ar[1]: 3})
# Run the pass on the dag from the input circuit
pass_ = StochasticSwap(coupling, layout, 20, 13)
after = pass_.run(dag)
# Verify the output of the pass matches our expectation
assert expected_dag == after
