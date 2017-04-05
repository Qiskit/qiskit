"""
Quantum teleportation example based on OPENQASM example.

Author: Andrew Cross
"""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, cx, u3, x, z

# define Program
# add circuit to Program
# add registers to circuit (or program?)

q = QuantumRegister("q", 3)
c0 = ClassicalRegister("c0", 1)
c1 = ClassicalRegister("c1", 1)
c2 = ClassicalRegister("c2", 1)
p = QuantumCircuit(q, c0, c1, c2)
q.u3(0.3, 0.2, 0.1, 0)
q.h(1)
q.cx(1, 2)
q.barrier()

q.cx(0, 1)
q.h(0)
p.measure(q[0], c0[0])
p.measure(q[1], c1[0])

q.z(2).c_if(c0, 1)
q.x(2).c_if(c1, 1)
p.measure(q[2], c2[0])

print(p.qasm())


# use methods instead - or have method as well
# c1 = a + b + c
# c2 = a + bp + c

# chemistry1 = make_variational_state + do_measurement_1
# chemistry2 = make_variational_state + do_measurement_2

# p.add_circuit(c1)
