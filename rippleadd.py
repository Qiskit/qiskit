"""
Ripple adder example based on OPENQASM example.

Author: Andrew Cross
"""
import qiskit as qk

# Work in progress
from qiskit.qasm import Qasm
import qiskit.unroll as unroll
import qiskit.mapper as mapper


def print_2x8(layout):
    """Print a 2x8 layout."""
    rev_layout = {b: a for a, b in layout.items()}
    print("")
    print("2x8 layout:")
    for i in range(8):
        qubit = ("q", i)
        if qubit in rev_layout:
            print("%s[%d] " % (rev_layout[qubit][0], rev_layout[qubit][1]),
                  end="")
        else:
            print("XXXX ", end="")
    print("")
    for i in range(8, 16):
        qubit = ("q", i)
        if qubit in rev_layout:
            print("%s[%d] " % (rev_layout[qubit][0], rev_layout[qubit][1]),
                  end="")
        else:
            print("XXXX ", end="")
    print("")


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
a = qk.QuantumRegister("a", n)
b = qk.QuantumRegister("b", n)
cin = qk.QuantumRegister("cin", 1)
cout = qk.QuantumRegister("cout", 1)
ans = qk.ClassicalRegister("ans", n+1)

# Build subcircuit to add a to b, storing result in b
adder_subcircuit = qk.QuantumCircuit(cin, a, b, cout)
majority(adder_subcircuit, cin[0], b[0], a[0])
for j in range(n-1):
    majority(adder_subcircuit, a[j], b[j+1], a[j+1])
adder_subcircuit.cx(a[n-1], cout[0])
for j in reversed(range(n-1)):
    unmajority(adder_subcircuit, a[j], b[j+1], a[j+1])
unmajority(adder_subcircuit, cin[0], b[0], a[0])

# Build the adder example
qc = qk.QuantumCircuit(cin, a, b, cout, ans)
qc.x(a[0])  # Set input a = 0...0001
qc.x(b)   # Set input b = 1...1111
qc += adder_subcircuit
for j in range(n):
    qc.measure(b[j], ans[j])  # Measure the output register
qc.measure(cout[0], ans[n])

######################################################################
# Work in progress
######################################################################

print("QuantumCircuit OPENQASM")
print("-----------------------")
print(qc.qasm())

u = unroll.Unroller(Qasm(data=qc.qasm()).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx"]))
u.execute()
C = u.be.C  # circuit directed graph object

print("")
print("Unrolled OPENQASM")
print("-----------------------")
print(C.qasm(qeflag=True))

print("")
print("size    = %d" % C.size())
print("depth   = %d" % C.depth())
print("width   = %d" % C.width())
print("bits    = %d" % C.num_cbits())
print("factors = %d" % C.num_tensor_factors())

# This is the 2 by 8
# Rewrite to use dict with integers (see quantum_optimization.py uder tools)
couplingdict = {0: [1, 8], 1: [2, 9], 2: [3, 10], 3: [4, 11], 4: [5, 12],
                5: [6, 13], 6: [7, 14], 7: [15], 8: [9], 9: [10], 10: [11],
                11: [12], 12: [13], 13: [14], 14: [15]}

coupling = mapper.Coupling(couplingdict)
print("")
print("2x8 coupling graph = \n%s" % coupling)

C_mapped, layout = mapper.swap_mapper(C, coupling)

print_2x8(layout)

print("")
print("SWAP mapped OPENQASM")
print("-----------------------")
print(C_mapped.qasm(qeflag=True))

u = unroll.Unroller(Qasm(data=C_mapped.qasm(qeflag=True)).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx"]))
u.execute()
C_mapped_unrolled = u.be.C

C_directions = mapper.direction_mapper(C_mapped_unrolled, coupling)

print("")
print("Direction mapped OPENQASM")
print("-----------------------")
print(C_directions.qasm(qeflag=True))

print("")
print("size    = %d" % C_directions.size())
print("depth   = %d" % C_directions.depth())
print("width   = %d" % C_directions.width())
print("bits    = %d" % C_directions.num_cbits())
print("factors = %d" % C_directions.num_tensor_factors())

# TODO: pass to remove CX.CX
# TODO: pass to simplify adjacent single qubit gate strings
