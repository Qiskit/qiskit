"""
Ripple adder example based on OPENQASM example.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister, ClassicalRegister, Program
from qiskit_sdk.extensions.standard import x, cx, ccx

n = 4

cin = QuantumRegister("cin", 1)
a = QuantumRegister("a", n)
b = QuantumRegister("b", n)
cout = QuantumRegister("cout", 1)
ans = ClassicalRegister("ans", n+1)
p = Program(cin, a, b, cout, ans)


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
majority(p, (cin, 0), (b, 0), (a, 0))
for j in range(n-1):
    majority(p, (a, j), (b, j+1), (a, j+1))
p.cx((a, n-1), (cout, 0))
for j in reversed(range(n-1)):
    unmajority(p, (a, j), (b, j+1), (a, j+1))
unmajority(p, (cin, 0), (b, 0), (a, 0))
for j in range(n):
    p.measure((b, j), (ans, j))
p.measure((cout, 0), (ans, n))

print(p.qasm())
