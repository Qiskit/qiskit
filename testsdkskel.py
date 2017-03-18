"""
Test the SDK skeleton.

Author: Andrew Cross
"""
from qiskit_sdk import QuantumRegister, ClassicalRegister, Program
from qiskit_sdk.extensions.standard import h, cx

# Issues:
# - How do we treat composites? - make an extension example of this
# - Get examples going with gate methods, like inverse

n = 5
q = QuantumRegister("q", n)
c = ClassicalRegister("c", n)
# q.h(0)  # this raises exception
p = Program(q, c)
pp = Program(q, c)  # it would be easy to bail here if that's preferred
q.h(0)  # this gets applied to q in p and pp
for i in range(n-1):
    q.cx(i, i+1)
for i in range(n):
    p.h((q, i))
    p.measure((q, i), (c, i))
    pp.measure((q, i), (c, i))
print(p.qasm())
print(pp.qasm())
