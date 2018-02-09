# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
import numpy as np
    
# importing the QISKit
from qiskit import QuantumCircuit, QuantumProgram
import Qconfig

# import tomography libary
import qiskit.tools.qcvv.tomography as tomo

# useful additional packages 
from qiskit.tools.visualization import plot_state, plot_histogram
from qiskit.tools.qi.qi import state_fidelity, concurrence, purity, outer

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config['url']) # set the APIToken and API url

# Creating registers
qr = Q_program.create_quantum_register('qr', 2)
cr = Q_program.create_classical_register('cr', 2)

# quantum circuit to make an entangled bell state 
bell = Q_program.create_circuit('bell', [qr], [cr])
bell.h(qr[0])
bell.cx(qr[0], qr[1])

# Construct state tomography set for measurement of qubits [0, 1] in the Pauli basis
bell_tomo_set = tomo.state_tomography_set([0, 1])

# Add the state tomography measurement circuits to the Quantum Program
bell_tomo_circuits = tomo.create_tomography_circuits(Q_program, 'bell', qr, cr, bell_tomo_set)
print('Created State tomography circuit labels:')
for c in bell_tomo_circuits:
    print(c)

# Use the local simulator
backend = 'local_qasm_simulator'

# Take 5000 shots for each measurement basis
shots = 5000

# Run the simulation
bell_tomo_result = Q_program.execute(bell_tomo_circuits, backend=backend, shots=shots)
print(bell_tomo_result)

bell_tomo_data = tomo.tomography_data(bell_tomo_result, 'bell', bell_tomo_set)

rho_fit = tomo.fit_tomography_data(bell_tomo_data)

# target state is (|00>+|11>)/sqrt(2)
target = np.array([1., 0., 0., 1.]/np.sqrt(2.))

# calculate fidelity, concurrence and purtity of fitted state
F_fit = state_fidelity(rho_fit, [0.707107, 0, 0, 0.707107])
con = concurrence(rho_fit)
pur = purity(rho_fit)
print('Fidelity =', F_fit)
print('concurrence = ', str(con))
print('purity = ', str(pur))


