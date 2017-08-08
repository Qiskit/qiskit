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
Demo the use of the InitializeGate class to prepare arbitrary pure states.
"""

import sys, os, math
from qiskit import QuantumProgram
from qiskit.extensions.quantum_initializer import InitializeGate
import Qconfig

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

###############################################################
# Make a quantum program for state initialization.
###############################################################
Q_SPECS = {
    "name": "Program-tutorial",
    "circuits": [{
        "name": "initializerCirc",
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
circuit = Q_program.get_circuit("initializerCirc")
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

circuit.initialize("QInit", desired_vector, [qr[0], qr[1], qr[2], qr[3]])

circuit.measure(qr[0], cr[0])
circuit.measure(qr[1], cr[1])
circuit.measure(qr[2], cr[2])
circuit.measure(qr[3], cr[3])

QASM_source = Q_program.get_qasm("initializerCirc")

print(QASM_source)

###############################################################
# Set the backend name and coupling map.
###############################################################
device = 'ibmqx2'
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}
circuits = ['initializerCirc']
myshots = 1024

###############################################################
# Set up the API and execute the program.
###############################################################
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])

# Desired vector
print("Desired probabilities...")
print(list(map(lambda x: format(abs(x * x), '.3f'), desired_vector)).__str__())

# Initialize on local simulator
result = Q_program.execute(circuits, backend='local_qasm_simulator', wait=2, timeout=240, shots=myshots)

print("Probabilities from simulator...[%s]" % result)
n_qubits_qureg = qr.size
print([format(result.get_counts("initializerCirc").get(format(i, '0' + n_qubits_qureg.__str__() + 'b'), 0) / myshots, '.3f') 
    for i in range(2 ** n_qubits_qureg)].__str__())

# Initialize on real backend (TODO: uncomment after reset is supported)
"""
result = Q_program.execute(circuits, backend=device, coupling_map=coupling_map, wait=2, timeout=240, shots=myshots)
print("Probabilities from device...[%s]" % result)
n_qubits_qureg = qr.size
n_qubits_device = Q_program.get_backend_configuration('ibmqx2').get('n_qubits', 0)
print([format(result.get_counts("initializerCirc").get(
    '0'+format(i, '0'*(n_qubits_device - n_qubits_qureg) + n_qubits_qureg.__str__() + 'b'), 0) / myshots,'.3f') 
    for i in range(2 ** n_qubits_qureg)].__str__())
    """
