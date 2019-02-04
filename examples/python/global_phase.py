from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute, BasicAer
from qiskit.transpiler import PassManager

import numpy as np

desired_vector = [-0.5, 0.5, 0.5, 0.5]
qr = QuantumRegister(2, "qr")
qc = QuantumCircuit(qr)
qc.initialize(desired_vector, [qr[0], qr[1]])

simulator='statevector_simulator'

job = execute(qc, BasicAer.get_backend(simulator), pass_manager=None)
result = job.result()
statevector = result.get_statevector()
print("State Vector (without global phase adjustment): ", statevector)

#Don't know why this doesn't work:
#phase_fix=qc.data[0]._get_unachieved_global_phase()
#print("Left-over phase:", phase_fix)
#Manually choose global phase to be -1
phase_fix=np.exp(np.complex(0,np.pi))

qc.globalphase(phase_fix)

job = execute(qc, BasicAer.get_backend(simulator), pass_manager=None)
result = job.result()
statevector = result.get_statevector()
print("State Vector (after global phase adjustment): ", statevector)

#The _get_unachieved_global_phase() doesn't work with the unitary simulator either:
#simulator='unitary_simulator'
#job = execute(qc, BasicAer.get_backend(simulator), pass_manager=None)
#result = job.result()
#unitary = result.get_unitary(qc)
#print("Circuit unitary:\n", unitary)
