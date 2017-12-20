# coding=utf-8
"""Demo the use of the Initialiser class to intialise a quantum register to an arbitrary complex vector (normalised)."""

import sys
from qiskit import QuantumProgram
from qiskit.extensions.quantum_initialiser import Initialiser
import math
# import Qconfig

sys.path.append(
    "../../")  # solve the relative dependencies if you clone QISKit from the Git repo and use like a global.

Q_SPECS = {
    "name": "Program-tutorial",
    "circuits": [{
        "name": "Circuit",
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
circuit = Q_program.get_circuit("Circuit")
quantum_r = Q_program.get_quantum_register("qr")
classical_r = Q_program.get_classical_register('cr')

desired_vector = [1 / math.sqrt(4) * complex(0, 1), 1 / (math.sqrt(8)) * complex(1, 0), 0, 0, 0, 0, 0, 0,
                  1 / (math.sqrt(8)) * complex(1, 0), 0, 0, 0, 0, 0, 1 / math.sqrt(4) * complex(1, 0),
                  1 / math.sqrt(4) * complex(1, 0)]

circuit.initialise("QInit",desired_vector, [quantum_r[0], quantum_r[1], quantum_r[2], quantum_r[3]])

circuit.measure(quantum_r[0], classical_r[0])
circuit.measure(quantum_r[1], classical_r[1])
circuit.measure(quantum_r[2], classical_r[2])
circuit.measure(quantum_r[3], classical_r[3])

QASM_source = Q_program.get_qasm("Circuit")

print(QASM_source)

device = 'local_qasm_simulator'
circuits = ['Circuit']

# Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) #set the APIToken and API url

#Q_program.compile(circuits, device)

result = Q_program.execute(circuits, backend=device, wait=2, timeout=240)

print(result)

print(Q_program.get_qasms(circuits))

# TODO: myshots = Q_program.__quantum_program["circuits"][name]["execution"][backend][shots]   ?

myshots = 1024
num_qubits = quantum_r.size
print("Quantum : " + [
    format(result.get_counts("Circuit").get(format(i, '0' + num_qubits.__str__() + 'b'), 0) / myshots, '.3f') for i
    in range(2 ** num_qubits)].__str__())
print("Original: " + list(map(lambda x: format(abs(x * x), '.3f'), desired_vector)).__str__())
