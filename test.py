from qiskit.circuit import *
import numpy as np
from qiskit._accelerate.circuit import draw

# wire for each clbit
# [10, 11, 11, 12, 12]

# clbit indices
# [0, 1, 2, 3, 4]

# index being passed
# [12, 12, 11]

# ind - clbit_map[clbit.first().index()]

# required output:
# 0, 1, 1

creg1 = ClassicalRegister(1, 'c1')
creg2 = ClassicalRegister(2, 'c2')
creg3 = ClassicalRegister(101, 'c3')
qreg1 = QuantumRegister(10, 'q1')
qc = QuantumCircuit(qreg1, creg1, creg2, creg3)
# qc.iswap(qreg1[9], qreg1[8])
qc.measure(qreg1[0], creg3[0])
qc.measure(qreg1[1], creg3[99])
qc.measure(qreg1[2], creg2[1])
creg = ClassicalRegister(4, 'creg')
qc.add_register(creg)
qc.measure(qreg1[3], creg[0])
print(draw(qc, cregbundle=True, mergewires =False))