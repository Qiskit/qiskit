"""
Test script for mapping quantum circuits.

This script takes a QASM file and produces output that shows the
process of mapping the circuit onto the coupling graph.
The final output is QASM source for a "mapped" version of the input
circuit.

The coupling graph and target basis are hard-coded into this script.

Author: Andrew Cross
        Jesus Perez <jesusper@us.ibm.com>
"""

import sys
import os

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
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


def print_qe5(layout):
    """Print a QE5 layout."""
    rev_layout = {b: a for a, b in layout.items()}
    print("")
    print("QE5 layout:")
    if ("q", 0) in rev_layout:
        q0 = "%s[%d]" % (rev_layout[("q", 0)][0], rev_layout[("q", 0)][1])
    else:
        q0 = "XXXX"
    if ("q", 1) in rev_layout:
        q1 = "%s[%d]" % (rev_layout[("q", 1)][0], rev_layout[("q", 1)][1])
    else:
        q1 = "XXXX"
    if ("q", 2) in rev_layout:
        q2 = "%s[%d]" % (rev_layout[("q", 2)][0], rev_layout[("q", 2)][1])
    else:
        q2 = "XXXX"
    if ("q", 3) in rev_layout:
        q3 = "%s[%d]" % (rev_layout[("q", 3)][0], rev_layout[("q", 3)][1])
    else:
        q3 = "XXXX"
    if ("q", 4) in rev_layout:
        q4 = "%s[%d]" % (rev_layout[("q", 4)][0], rev_layout[("q", 4)][1])
    else:
        q4 = "XXXX"
    print("%s    %s" % (q0, q1))
    print("    %s" % q4)
    print("%s    %s" % (q2, q3))
    print("")


def make_unrolled_circuit(fname, basis):
    """
    Create a graph representation of the QASM circuit.

    basis is a comma separated list of operation names.
    The circuit is unrolled to gates in the basis.
    """
    ast = Qasm(filename=fname).parse()
    u = unroll.Unroller(ast, unroll.CircuitBackend(basis.split(",")))
    u.execute()
    return u.backend.circuit


def make_unrolled_circuit_from_data(dat, basis):
    """
    Create a graph representation of the QASM circuit.

    basis is a comma separated list of operation names.
    The circuit is unrolled to gates in the basis.
    """
    ast = Qasm(data=dat).parse()
    u = unroll.Unroller(ast, unroll.CircuitBackend(basis.split(",")))
    u.execute()
    return u.backend.circuit


if len(sys.argv) < 2:
    print("mapper.py <qasm>\n")
    print("  qasm = main circuit")
    print("")
    print("Generates a new \"mapped\" circuit matching the coupling.")
    sys.exit(1)

# This is the QE basis
basis = "u1,u2,u3,cx,id"

layout_type = "qe5"  # "qe5" or "2x8"

# Initialize coupling string
if layout_type == "qe5":
    # This is the star graph
    couplingdict = {0: [4], 1: [4], 2: [4], 3: [4]}
elif layout_type == "2x8":
    # This is the 2 by 8
    couplingdict = {0: [1, 8], 1: [2, 9], 2: [3, 10], 3: [4, 11], 4: [5, 12],
                    5: [6, 13], 6: [7, 14], 7: [15], 8: [9], 9: [10], 10: [11],
                    11: [12], 12: [13], 13: [14], 14: [15]}
else:
    print("bad layout type")
    sys.exit(1)

# First, unroll the input circuit
c = make_unrolled_circuit(sys.argv[1], basis)

# Second, create the coupling graph
coupling = mapper.Coupling(couplingdict)
print("coupling = \n%s" % coupling)

# Third, do the mapping
c_prime, layout = mapper.swap_mapper(c, coupling)
print("c_prime.qasm() = \n%s" % c_prime.qasm(qeflag=True))

if layout_type == "qe5":
    print_qe5(layout)
elif layout_type == "2x8":
    print_2x8(layout)

# Three', expand swaps into cx gates
c_prime = make_unrolled_circuit_from_data(c_prime.qasm(qeflag=True),
                                          "cx,u1,u2,u3,id")

# Fourth, do direction mapping
c_dblp = mapper.direction_mapper(c_prime, coupling, verbose=True)
print("c_dblp.qasm() = \n%s" % c_dblp.qasm(qeflag=True))

# Unroll again
c_final = make_unrolled_circuit_from_data(c_dblp.qasm(qeflag=True),
                                          "cx,u1,u2,u3,id")
print("c_final.qasm() = \n%s" % c_final.qasm(qeflag=True))
