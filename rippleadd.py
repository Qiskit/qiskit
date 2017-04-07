"""
Ripple adder example based on OPENQASM example.

Author: Andrew Cross
"""
# one import statement here would be ideal
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import x, cx, ccx

from qiskit.qasm import Qasm
import qiskit.unroll as unroll


def majority(p, a, b, c):
    """Majority gate."""
    p.cx(c, b)
    p.cx(c, a)
    p.ccx(a, b, c)


def unmajority(p, a, b, c):
    """Unmajority gate."""
    p.ccx(a, b, c)
    p.cx(c, a)
    p.cx(a, b)

# p = QP()
# c = p.C()
# c.add(QR())
# c.add(b)
# qr = p.c.qr()
# cr = p.c.cr()
# p = QuantumCircuit(QuantumRegister("cin", 1),
#                     QuantumRegister("a", n),
#                     ...)
# p.regs["cin"].x()
# something like p = Program(c1, c2, c3)
# circ.QuantumRegister("a", n)

n = 8

a = QuantumRegister("a", n)
b = QuantumRegister("b", n)
cin = QuantumRegister("cin", 1)
cout = QuantumRegister("cout", 1)
ans = ClassicalRegister("ans", n+1)

adder_subcircuit = QuantumCircuit(cin, a, b, cout)

# Build subcircuit to add a to b, storing result in b
majority(adder_subcircuit, cin[0], b[0], a[0])
for j in range(n-1):
    majority(adder_subcircuit, a[j], b[j+1], a[j+1])
adder_subcircuit.cx(a[n-1], cout[0])
for j in reversed(range(n-1)):
    unmajority(adder_subcircuit, a[j], b[j+1], a[j+1])
unmajority(adder_subcircuit, cin[0], b[0], a[0])

qc = QuantumCircuit(cin, a, b, cout, ans)

qc.x(a[0])  # Set input a = 0...0001
qc.x(b)   # Set input b = 1...1111
qc += adder_subcircuit
for j in range(n):
    qc.measure(b[j], ans[j])  # Measure the output register
qc.measure(cout[0], ans[n])

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

# print("")
# print("Unrolled OPENQASM")
# print("-----------------------")
# print(C.qasm(qeflag=True))
