# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example use of the initialize gate to prepare arbitrary pure states.

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, BasicAer


###############################################################
# Make a quantum circuit for state initialization.
###############################################################
qr = QuantumRegister(4, "qr")
cr = ClassicalRegister(4, 'cr')
circuit = QuantumCircuit(qr, cr, name="initializer_circ")

desired_vector = [
    1 / math.sqrt(4) * complex(0, 1),
    1 / math.sqrt(8) * complex(1, 0),
    0,
    0,
    0,
    0,
    0,
    0,
    1 / math.sqrt(8) * complex(1, 0),
    1 / math.sqrt(8) * complex(0, 1),
    0,
    0,
    0,
    0,
    1 / math.sqrt(4) * complex(1, 0),
    1 / math.sqrt(8) * complex(1, 0)]

circuit.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])

circuit.measure(qr[0], cr[0])
circuit.measure(qr[1], cr[1])
circuit.measure(qr[2], cr[2])
circuit.measure(qr[3], cr[3])

print(circuit.qasm())

###############################################################
# Execute on a simulator backend.
###############################################################
shots = 10000

# Desired vector
print("Desired probabilities...")
print(str(list(map(lambda x: format(abs(x * x), '.3f'), desired_vector))))

# Initialize on local simulator
sim_backend = BasicAer.get_backend('qasm_simulator')
job = execute(circuit, sim_backend, shots=shots)
result = job.result()

print("Probabilities from simulator...[%s]" % result)
n_qubits_qureg = qr.size
counts = result.get_counts(circuit)

qubit_strings = [format(i, '0%sb' % n_qubits_qureg) for
                 i in range(2 ** n_qubits_qureg)]
print([format(counts.get(s, 0) / shots, '.3f') for
       s in qubit_strings])
