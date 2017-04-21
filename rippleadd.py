"""
Ripple adder example based on OPENQASM example.

Author: Andrew Cross
"""
# one import statement here would be ideal
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import x, cx, ccx

from qiskit.qasm import Qasm
import qiskit.unroll as unroll
import qiskit.mapper as mapper


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


n = 4

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
print("SWAP mapped OPENQASM")
print("-----------------------")
print(C_mapped.qasm(qeflag=True))

C_directions = mapper.direction_mapper(C_mapped, coupling, verbose=True)

print("")
print("Direction mapped OPENQASM")
print("-----------------------")
print(C_directions.qasm(qeflag=True))
