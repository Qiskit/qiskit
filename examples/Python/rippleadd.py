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
# Work in progress to map the QuantumCircuit to the 2x8 device
######################################################################

print("QuantumCircuit OPENQASM")
print("-----------------------")
print(qc.qasm())

######################################################################
# First pass: expand subroutines to a basis of 1 and 2 qubit gates.
######################################################################

u = unroll.Unroller(Qasm(data=qc.qasm()).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx"]))
u.execute()
C = u.backend.circuit  # circuit directed graph object

print("Unrolled OPENQASM to [u1, u2, u3, cx] basis")
print("-------------------------------------------")
# print(C.qasm(qeflag=True))

# print("")
print("size    = %d" % C.size())
print("depth   = %d" % C.depth())
print("width   = %d" % C.width())
print("bits    = %d" % C.num_cbits())
print("factors = %d" % C.num_tensor_factors())

######################################################################
# Second pass: choose a layout on the coupling graph and add SWAPs.
######################################################################

# This is the 2 by 8 array of 16 qubits
couplingdict = {0: [1, 8], 1: [2, 9], 2: [3, 10], 3: [4, 11], 4: [5, 12],
                5: [6, 13], 6: [7, 14], 7: [15], 8: [9], 9: [10], 10: [11],
                11: [12], 12: [13], 13: [14], 14: [15]}

coupling = mapper.Coupling(couplingdict)
C_mapped, layout = mapper.swap_mapper(C, coupling)

print("")
print("Initial qubit positions on 2x8 layout")
print("-------------------------------------------")
print_2x8(layout)

print("")
print("Inserted SWAP gates for 2x8 layout")
print("-------------------------------------------")
# print(C_mapped.qasm(qeflag=True))

# print("")
print("size    = %d" % C_mapped.size())
print("depth   = %d" % C_mapped.depth())
print("width   = %d" % C_mapped.width())
print("bits    = %d" % C_mapped.num_cbits())
print("factors = %d" % C_mapped.num_tensor_factors())

######################################################################
# Third pass: expand SWAP subroutines and adjust cx gate orientations.
######################################################################

u = unroll.Unroller(Qasm(data=C_mapped.qasm(qeflag=True)).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx"]))
u.execute()
C_mapped_unrolled = u.backend.circuit
C_directions = mapper.direction_mapper(C_mapped_unrolled, coupling)

print("")
print("Changed CNOT directions")
print("-------------------------------------------")
# print(C_directions.qasm(qeflag=True))

# print("")
print("size    = %d" % C_directions.size())
print("depth   = %d" % C_directions.depth())
print("width   = %d" % C_directions.width())
print("bits    = %d" % C_directions.num_cbits())
print("factors = %d" % C_directions.num_tensor_factors())

######################################################################
# Fourth pass: collect runs of cx gates and cancel them.
######################################################################
# We assume there are no double edges in the connectivity graph, so
# we don't need to check the direction of the cx gates in a run.
runs = C_directions.collect_runs(["cx"])
for r in runs:
    if len(r) % 2 == 0:
        for n in r:
            C_directions._remove_op_node(n)
    else:
        for j in range(len(r)-1):
            C_directions._remove_op_node(r[j])

print("")
print("Cancelled redundant CNOT gates")
print("-------------------------------------------")
# print(C_directions.qasm(qeflag=True))

# print("")
print("size    = %d" % C_directions.size())
print("depth   = %d" % C_directions.depth())
print("width   = %d" % C_directions.width())
print("bits    = %d" % C_directions.num_cbits())
print("factors = %d" % C_directions.num_tensor_factors())

######################################################################
# Fifth pass: expand single qubit gates to u1, u2, u3 and simplify.
######################################################################

u = unroll.Unroller(Qasm(data=C_directions.qasm(qeflag=True)).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx"]))
u.execute()
C_directions_unrolled = u.backend.circuit

runs = C_directions_unrolled.collect_runs(["u1", "u2", "u3"])
for run in runs:
    qname = C_directions_unrolled.multi_graph.node[run[0]]["qargs"][0]
    print("%s, %d, %s[%d]: " % (run, len(run), qname[0], qname[1]))
    for node in run:
        nd = C_directions_unrolled.multi_graph.node[node]
        assert nd["condition"] is None, "internal error"
        assert len(nd["qargs"]) == 1, "internal error"
        assert nd["qargs"][0] == qname, "internal error"
        print(" %s %s" % (nd["name"], nd["params"]))

# Single qubit gate composition rules:
# --------------------------------------
# u1(lambda1) * u1(lambda2) = u1(lambda1 + lambda2)
# u1(lambda1) * u2(phi2, lambda2) = u2(phi2 + lambda1, phi2)
# u2(phi1, lambda1) * u1(lambda2) = u2(phi1, lambda1 + lambda2)
# u1(lambda1) * u3(theta2, phi2, lambda2) = u3(theta2, phi2 + lambda1, lambda2)
# u3(theta1, phi1, lambda1) * u1(lambda2) = u3(theta1, phi1, lambda1 + lambda2)
#
# Using Ry(pi/2).Rz(2*lambda).Ry(pi/2) = Rz(pi/2).Ry(pi-2*lambda).Rz(pi/2),
# u2(phi1, lambda1) * u2(phi2, lambda2) = u3(pi - lambda1 - phi2, phi1 + pi/2, lambda2 + pi/2)
#
# For composing u3's or u2's with u3's, use u2(phi, lambda) = u3(pi/2, phi, lambda)
# together with the qiskit.mapper.compose_u3 method.

# TODO: complete this pass
# TODO: add Circuit method to tally each type of gate so we can see costs
# TODO: test on examples using simulator

# TODO: simple unit tests for qasm, mapper, unroller, circuit
