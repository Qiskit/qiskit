"""
Test script for localization of quantum circuits.

This script takes a QASM file and produces output that shows the
process of finding SWAP gates to map the circuit onto the coupling graph.
The final output is QASM source for a "localized" version of the input
circuit.

The coupling graph and target basis are hard-coded into this script.

Author: Andrew Cross
"""

import sys
import qx_sdk.Qasm as Qasm
import qx_sdk.QasmBackends as QasmBackends
import qx_sdk.QasmInterpreters as QasmInterpreters
import qx_sdk.Localize as Localize


def make_unrolled_circuit(fname, basis):
    """
    Create a graph representation of the QASM circuit.

    The circuit is unrolled to gates in the input basis.
    """
    q = Qasm.Qasm(filename=fname)
    ast = q.parse()
    be = QasmBackends.CircuitBackend()
    be.set_basis(basis.split(","))
    u = QasmInterpreters.Unroller(ast, be)
    u.execute()
    return be.cg


# Check the command line for a QASM file name
if len(sys.argv) < 2:
    print("localize.py <qasm>\n")
    print("  qasm = main circuit")
    print("")
    print("Generates a new \"localized\" circuit matching the coupling.")
    sys.exit(1)

# This is the QE basis
basis = "u1,u2,u3,cx"

# This is the star graph
couplingstr = "q,0:q,4;q,1:q,4;q,2:q,4;q,3:q,4"

# First, unroll the input circuit
c = make_unrolled_circuit(sys.argv[1], basis)

# Second, create the coupling graph
coupling = Localize.CouplingGraph(couplingstr)

print("CouplingGraph is = \n%s" % coupling)

if not coupling.connected():
    print("Coupling graph must be connected")
    sys.exit(1)

print("input circuit is = \n%s" % c.qasm())
print("circuit depth = %d" % c.depth())

# Here down is hacking for now

coupling.compute_distance()
for q1 in coupling.qubits.keys():
    for q2 in coupling.qubits.keys():
        print("%s[%d] -> %s[%d]: %f" % (q1[0], q1[1], q2[0], q2[1],
                                        coupling.distance(q1, q2)))

l = c.layers()
print("len(l) = %d" % len(l))
for i in range(len(l)):
    print("i = %d" % i)
    print("layer = \n%s" % l[i]["graph"].qasm())
    print("partition = %s" % l[i]["partition"])
