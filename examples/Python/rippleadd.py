"""
Ripple adder example based on OPENQASM example.

Author: Andrew Cross
"""
import math

import networkx as nx

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

# TODO: Put in new input, make and compile two circuits, put compile in method

######################################################################
# Map the QuantumCircuit to the 2x8 device
######################################################################

print("QuantumCircuit OPENQASM")
print("-----------------------")
print(qc.qasm())

# Unroll this now just for the purpose of gate counting
u = unroll.Unroller(Qasm(data=qc.qasm()).parse(),
                    unroll.CircuitBackend(["cx", "x", "ccx"]))
u.execute()
C = u.backend.circuit

print("")
print("size    = %d" % C.size())
print("depth   = %d" % C.depth())
print("width   = %d" % C.width())
print("bits    = %d" % C.num_cbits())
print("factors = %d" % C.num_tensor_factors())
print("operations:")
for key, val in C.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# First pass: expand subroutines to a basis of 1 and 2 qubit gates.
######################################################################

u = unroll.Unroller(Qasm(data=qc.qasm()).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx", "id"]))
u.execute()
C = u.backend.circuit

print("")
print("Unrolled OPENQASM to QX basis")
print("-------------------------------------------")
# print(C.qasm(qeflag=True))

# print("")
print("size    = %d" % C.size())
print("depth   = %d" % C.depth())
print("width   = %d" % C.width())
print("bits    = %d" % C.num_cbits())
print("factors = %d" % C.num_tensor_factors())
print("operations:")
for key, val in C.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# Second pass: choose a layout on the coupling graph and add SWAPs.
######################################################################

# This is a 2 by 8 array of 16 qubits
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
print("operations:")
for key, val in C_mapped.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# Third pass: expand SWAP subroutines and adjust cx gate orientations.
######################################################################

u = unroll.Unroller(Qasm(data=C_mapped.qasm(qeflag=True)).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx", "id"]))
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
print("operations:")
for key, val in C_directions.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# Fourth pass: collect runs of cx gates and cancel them.
######################################################################
# We assume there are no double edges in the connectivity graph, so
# we don't need to check the direction of the cx gates in a run.
# BUG: Remove this assumption - won't hold for all-all connections.
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
print("operations:")
for key, val in C_directions.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# Fifth pass: expand single qubit gates to u1, u2, u3 and simplify.
######################################################################

u = unroll.Unroller(Qasm(data=C_directions.qasm(qeflag=True)).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx", "id"]))
u.execute()
C_directions_unrolled = u.backend.circuit

runs = C_directions_unrolled.collect_runs(["u1", "u2", "u3", "id"])
for run in runs:
    qname = C_directions_unrolled.multi_graph.node[run[0]]["qargs"][0]
    right_name = "u1"
    right_parameters = (0.0, 0.0, 0.0)  # (theta, phi, lambda)
    for node in run:
        nd = C_directions_unrolled.multi_graph.node[node]
        assert nd["condition"] is None, "internal error"
        assert len(nd["qargs"]) == 1, "internal error"
        assert nd["qargs"][0] == qname, "internal error"
        left_name = nd["name"]
        assert left_name in ["u1", "u2", "u3", "id"], "internal error"
        if left_name == "u1":
            left_parameters = (0.0, 0.0, float(nd["params"][0]))
        elif left_name == "u2":
            left_parameters = (math.pi/2, float(nd["params"][0]),
                               float(nd["params"][1]))
        elif left_name == "u3":
            left_parameters = tuple(map(float, nd["params"]))
        else:
            left_name = "u1"  # replace id with u1
            left_parameters = (0.0, 0.0, 0.0)
        # Compose gates
        name_tuple = (left_name, right_name)
        if name_tuple == ("u1", "u1"):
            # u1(lambda1) * u1(lambda2) = u1(lambda1 + lambda2)
            right_parameters = (0.0, 0.0, right_parameters[2] +
                                left_parameters[2])
        elif name_tuple == ("u1", "u2"):
            # u1(lambda1) * u2(phi2, lambda2) = u2(phi2 + lambda1, lambda2)
            right_parameters = (math.pi/2, right_parameters[1] +
                                left_parameters[2], right_parameters[2])
        elif name_tuple == ("u2", "u1"):
            # u2(phi1, lambda1) * u1(lambda2) = u2(phi1, lambda1 + lambda2)
            right_name = "u2"
            right_parameters = (math.pi/2, left_parameters[1],
                                right_parameters[2] + left_parameters[2])
        elif name_tuple == ("u1", "u3"):
            # u1(lambda1) * u3(theta2, phi2, lambda2) =
            #     u3(theta2, phi2 + lambda1, lambda2)
            right_parameters = (right_parameters[0], right_parameters[1] +
                                left_parameters[2], right_parameters[2])
        elif name_tuple == ("u3", "u1"):
            # u3(theta1, phi1, lambda1) * u1(lambda2) =
            #     u3(theta1, phi1, lambda1 + lambda2)
            right_name = "u3"
            right_parameters = (left_parameters[0], left_parameters[1],
                                right_parameters[2] + left_parameters[2])
        elif name_tuple == ("u2", "u2"):
            # Using Ry(pi/2).Rz(2*lambda).Ry(pi/2) =
            #    Rz(pi/2).Ry(pi-2*lambda).Rz(pi/2),
            # u2(phi1, lambda1) * u2(phi2, lambda2) =
            #    u3(pi - lambda1 - phi2, phi1 + pi/2, lambda2 + pi/2)
            right_name = "u3"
            right_parameters = (math.pi - left_parameters[2] -
                                right_parameters[1], left_parameters[1] +
                                math.pi/2, right_parameters[2] +
                                math.pi/2)
        else:
            # For composing u3's or u2's with u3's, use
            # u2(phi, lambda) = u3(pi/2, phi, lambda)
            # together with the qiskit.mapper.compose_u3 method.
            right_name = "u3"
            right_parameters = mapper.compose_u3(left_parameters[0],
                                                 left_parameters[1],
                                                 left_parameters[2],
                                                 right_parameters[0],
                                                 right_parameters[1],
                                                 right_parameters[2])
        # Here down, when we simplify, we add f(theta) to lambda to correct
        # the global phase when f(theta) is 2*pi. This isn't necessary but the
        # other steps preserve the global phase, so we continue to do so.
        epsilon = 1e-9  # for comparison with zero
        # Y rotation is 0 mod 2*pi, so the gate is a u1
        if abs(right_parameters[0] % 2.0*math.pi) < epsilon \
           and right_name != "u1":
            right_name = "u1"
            right_parameters = (0.0, 0.0, right_parameters[1] +
                                right_parameters[2] +
                                right_parameters[0])
        # Y rotation is pi/2 or -pi/2 mod 2*pi, so the gate is a u2
        if right_name == "u3":
            # theta = pi/2 + 2*k*pi
            if abs((right_parameters[0] - math.pi/2) % 2.0*math.pi) < epsilon:
                right_name = "u2"
                right_parameters = (math.pi/2, right_parameters[1],
                                    right_parameters[2] +
                                    (right_parameters[0] - math.pi/2))
            # theta = -pi/2 + 2*k*pi
            if abs((right_parameters[0] + math.pi/2) % 2.0*math.pi) < epsilon:
                right_name = "u2"
                right_parameters = (math.pi/2, right_parameters[1] + math.pi,
                                    right_parameters[2] - math.pi +
                                    (right_parameters[0] + math.pi/2))
        # u1 and lambda is 0 mod 4*pi so gate is nop
        if right_name == "u1" and \
           abs(right_parameters[2] % 4.0*math.pi) < epsilon:
            right_name = "nop"
    # Replace the data of the first node in the run
    new_params = []
    if right_name == "u1":
        new_params.append(right_parameters[2])
    if right_name == "u2":
        new_params = [right_parameters[1], right_parameters[2]]
    if right_name == "u3":
        new_params = list(right_parameters)
    nx.set_node_attributes(C_directions_unrolled.multi_graph, 'name',
                           {run[0]: right_name})
    nx.set_node_attributes(C_directions_unrolled.multi_graph, 'params',
                           {run[0]: tuple(map(str, new_params))})
    # Delete the other nodes in the run
    for node in run[1:]:
        C_directions_unrolled._remove_op_node(node)
    if right_name == "nop":
        C_directions_unrolled._remove_op_node(run[0])

print("")
print("Cancelled redundant single qubit gates")
print("-------------------------------------------")
print(C_directions_unrolled.qasm(qeflag=True))

print("")
print("size    = %d" % C_directions_unrolled.size())
print("depth   = %d" % C_directions_unrolled.depth())
print("width   = %d" % C_directions_unrolled.width())
print("bits    = %d" % C_directions_unrolled.num_cbits())
print("factors = %d" % C_directions_unrolled.num_tensor_factors())
print("operations:")
for key, val in C_directions_unrolled.count_ops().items():
    print("  %s: %d" % (key, val))

# TODO: put each step into a compile() method, clean up

# TODO: test on examples using simulator
# TODO: simple unit tests for qasm, mapper, unroller, circuit
