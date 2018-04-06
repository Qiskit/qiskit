# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Example use of the initialize gate to prepare arbitrary pure states.

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

import math
from qiskit import QuantumProgram
from qiskit.tools.visualization import plot_circuit
import Qconfig


###############################################################
# Make a quantum program for state initialization.
###############################################################
Q_SPECS = {
    "name": "Program-tutorial",
    "circuits": [{
        "name": "initializer_circ",
        "quantum_registers": [{
            "name": "qr",
            "size": 4
        }],
        "classical_registers": [{
            "name": "cr",
            "size": 4
        }]}],
}
Q_program = QuantumProgram(specs=Q_SPECS)
circuit = Q_program.get_circuit("initializer_circ")
qr = Q_program.get_quantum_register("qr")
cr = Q_program.get_classical_register('cr')

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

QASM_source = Q_program.get_qasm("initializer_circ")

print(QASM_source)
plot_circuit(circuit)

###############################################################
# Set the backend name and coupling map.
###############################################################
device = 'ibmqx2'
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}
circuits = ['initializer_circ']
shots = 1024

###############################################################
# Set up the API and execute the program.
###############################################################
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])

# Desired vector
print("Desired probabilities...")
print(str(list(map(lambda x: format(abs(x * x), '.3f'), desired_vector))))

# Initialize on local simulator
result = Q_program.execute(circuits,
                           backend='local_qasm_simulator',
                           wait=2, timeout=240, shots=shots)

print("Probabilities from simulator...[%s]" % result)
n_qubits_qureg = qr.size
counts = result.get_counts("initializer_circ")

qubit_strings = [format(i, '0%sb' % n_qubits_qureg) for
                 i in range(2 ** n_qubits_qureg)]
print([format(counts.get(s, 0) / shots, '.3f') for
       s in qubit_strings])
