"""
Ripple adder example based on OPENQASM example.

Author: Andrew Cross
"""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import x, cx, ccx

from qiskit.qasm import Qasm
import qiskit.unroll as unroll

n = 8

cin = QuantumRegister("cin", 1)
a = QuantumRegister("a", n)
b = QuantumRegister("b", n)
cout = QuantumRegister("cout", 1)
ans = ClassicalRegister("ans", n+1)

p = QuantumCircuit(cin, a, b, cout, ans)


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


# Set the input states
a.x(0)  # a = 0...0001
b.x()   # b = 1...1111

# Add a to b, storing result in b
majority(p, cin[0], b[0], a[0])
for j in range(n-1):
    majority(p, a[j], b[j+1], a[j+1])
p.cx(a[n-1], cout[0])
for j in reversed(range(n-1)):
    unmajority(p, a[j], b[j+1], a[j+1])
unmajority(p, cin[0], b[0], a[0])
for j in range(n):
    p.measure(b[j], ans[j])
p.measure(cout[0], ans[n])

print("QuantumCircuit OPENQASM")
print("-----------------------")
print(p.qasm())

basis = "u1,u2,u3,cx"  # QE target basis
ast = Qasm(data=p.qasm()).parse()
u = unroll.Unroller(ast, unroll.CircuitBackend(basis.split(",")))
u.execute()
C = u.be.C  # circuit directed graph object

#  TODO: do circuit optimizations, mapping
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
