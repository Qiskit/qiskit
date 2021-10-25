from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.aqua.operators import SummedOp, StateFn, Z, I, ListOp
from qiskit.aqua.operators.gradients import NaturalGradient
from qiskit.circuit.library import EfficientSU2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import ZGate, YGate, CZGate, CXGate
from qiskit.circuit import Parameter


# qr = QuantumRegister(2)
# qr_v = QuantumRegister(1)
# cr = ClassicalRegister(2)
# qc = QuantumCircuit(qr, cr)
# v = YGate(label='V')
# u = YGate(label='U')
# w = YGate(label='W')
# r = YGate(label='R')
# cv = v.control(1)
# cu = u.control(1)
# qc.append(cv, qr)
# qc.append(w, [qr[1]])
# qc.x(qr[0])
# qc.append(cu, qr)
# qc.x(qr[0])
# qc.append(r, [qr[1]])
# qc.h(qr[0])
# qc.measure(qr[0], cr[0])
# qc.measure(qr[1], cr[1])
# qc.draw(output='mpl', filename='exp_val_with_H.pdf')
#
# qr = QuantumRegister(3)
# cr = ClassicalRegister(2)
# qc = QuantumCircuit(qr, cr)
# v = YGate(label='V')
# u = YGate(label='U')
# w = YGate(label='W')
# r = YGate(label='R')
# cv = v.control(1)
# cu = u.control(1)
# qc.append(cv, [qr[1], [qr[2]]])
# qc.append(w, [qr[2]])
# qc.append(cu, [qr[0], [qr[2]]])
# qc.append(r, [qr[2]])
# qc.h(qr[1])
# qc.cz(qr[0], qr[1])
# qc.h(qr[0])
# qc.measure(qr[0], cr[0])
# qc.measure(qr[2], cr[1])
# qc.draw(output='mpl', filename='hess_with_H.pdf')

# qr = QuantumRegister(2)
# qr_v = QuantumRegister(1)
# cr = ClassicalRegister(1)
# qc = QuantumCircuit(qr, cr)
# v = YGate(label='V')
# u = YGate(label='U')
# w = YGate(label='W')
# r = YGate(label='R')
# cv = v.control(1)
# cu = u.control(1)
# qc.append(cv, qr)
# qc.append(w, [qr[1]])
# qc.x(qr[0])
# qc.append(cu, qr)
# qc.x(qr[0])
# qc.h(qr[0])
# # qc.append(r, [qr[1]])
# qc.measure(qr[0], cr[0])
# qc.draw(output='mpl', filename='exp_val.pdf')

# ra = RealAmplitudes(num_qubits=4, reps=1, entanglement=[[3,0], [2,3], [1, 2], [0, 1]])
# import numpy as np
# params = np.zeros(8)
# params[-4] = np.pi/2
# params[-3] = np.pi/2
# ra = ra.assign_parameters(dict(zip(ra.ordered_parameters, params)))
# qr = QuantumRegister(4)
# # cr = ClassicalRegister(1)
# qc = QuantumCircuit(qr)
# qc = qc.compose(ra, qr)
# qc.cx(qr[0], qr[2])
# qc.cx(qr[1], qr[3])
# qc.draw(output='mpl', filename='QBM_discr_ansatz.pdf')

ra = EfficientSU2(num_qubits=6, reps=2, entanglement='sca')
import numpy as np
params = np.zeros(36)
params[-10] = np.pi/2
params[-11] = np.pi/2
params[-12] = np.pi/2
ra = ra.assign_parameters(dict(zip(ra.ordered_parameters, params)))
qr = QuantumRegister(6)
# cr = ClassicalRegister(1)
qc = QuantumCircuit(qr)
qc = qc.compose(ra, qr)
qc.cx(qr[0], qr[3])
qc.cx(qr[1], qr[4])
qc.cx(qr[2], qr[5])
qc.draw(output='mpl', filename='QBM_ghz_ansatz.pdf')
# qc = RealAmplitudes(num_qubits=3, reps=1, entanglement='linear',
#                     parameter_prefix=r'$\omega$')
# qc.entanglement_blocks = CZGate
# params = qc.ordered_parameters
# new_params = [Parameter(r'$\omega_0$'), Parameter(r'$\omega_1$'), Parameter(r'$\omega_2$'),
#               Parameter(r'$\omega_3$'), Parameter(r'$\omega_4$'),
#               Parameter(r'$\omega_5$')]
#
# qc.ordered_parameters = new_params
# qc.draw(output='mpl', filename='RealAmplitudes_linear.pdf')

# qc = EfficientSU2(num_qubits=3, reps=1, parameter_prefix=r'$\omega$')
#
# new_params = [Parameter(r'$\omega_0$'), Parameter(r'$\omega_1$'), Parameter(r'$\omega_2$'),
#               Parameter(r'$\omega_3$'), Parameter(r'$\omega_4$'),
#               Parameter(r'$\omega_5$'), Parameter(r'$\omega_6$'), Parameter(r'$\omega_7$'),
#               Parameter(r'$\omega_8$'),
#               Parameter(r'$\omega_9$'), Parameter(r'$\omega_{10}$'),
#               Parameter(r'$\omega_{11}$')]
#
# qc.ordered_parameters = new_params
# qc.draw(output='mpl', filename='EfficientSU2.pdf')