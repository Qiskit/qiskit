"""
Quantum teleportation example based on OPENQASM example.

Author: Andrew Cross
"""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import barrier, h, cx, u3, x, z

from qiskit.qasm import Qasm
import qiskit.unroll as unroll
import qiskit.mapper as mapper


q = QuantumRegister("q", 3)
c0 = ClassicalRegister("c0", 1)
c1 = ClassicalRegister("c1", 1)
c2 = ClassicalRegister("c2", 1)
qc = QuantumCircuit(q, c0, c1, c2)
qc.u3(0.3, 0.2, 0.1, q[0])
qc.h(q[1])
qc.cx(q[1], q[2])
qc.barrier(q)

qc.cx(q[0], q[1])
qc.h(q[0])
qc.measure(q[0], c0[0])
qc.measure(q[1], c1[0])

qc.z(q[2]).c_if(c0, 1)
qc.x(q[2]).c_if(c1, 1)
qc.measure(q[2], c2[0])

######################################################################

print("QuantumCircuit OPENQASM")
print("-----------------------")
print(qc.qasm())

u = unroll.Unroller(Qasm(data=qc.qasm()).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx"]))
u.execute()
C = u.be.C  # circuit directed graph object

print("")
print("size    = %d" % C.size())
print("depth   = %d" % C.depth())
print("width   = %d" % C.width())
print("bits    = %d" % C.num_cbits())
print("factors = %d" % C.num_tensor_factors())

print("")
print("Unrolled OPENQASM")
print("-----------------------")
print(C.qasm(qeflag=True))

# This is the 2 by 8
couplingstr = "q,0:q,1;q,1:q,2;q,2:q,3;q,3:q,4;q,4:q,5;q,5:q,6;q,6:q,7" + \
              ";q,8:q,9;q,9:q,10;q,10:q,11;q,11:q,12;q,12:q,13;q,13:q,14" + \
              ";q,14:q,15" + \
              ";q,0:q,8;q,1:q,9;q,2:q,10;q,3:q,11;q,4:q,12;q,5:q,13" + \
              ";q,6:q,14;q,7:q,15"

coupling = mapper.Coupling(couplingstr)
print("")
print("2x8 coupling graph = \n%s" % coupling)

C_mapped, layout = mapper.swap_mapper(C, coupling)
rev_layout = {b: a for a, b in layout.items()}

print("")
print("2x8 layout:")
for i in range(8):
    qubit = ("q", i)
    if qubit in rev_layout:
        print("%s[%d] " % (rev_layout[qubit][0], rev_layout[qubit][1]), end="")
    else:
        print("XXXX ", end="")
print("")
for i in range(8, 16):
    qubit = ("q", i)
    if qubit in rev_layout:
        print("%s[%d] " % (rev_layout[qubit][0], rev_layout[qubit][1]), end="")
    else:
        print("XXXX ", end="")
print("")

print("")
print("Mapped OPENQASM")
print("-----------------------")
print(C_mapped.qasm(qeflag=True))
