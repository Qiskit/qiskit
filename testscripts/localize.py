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
sys.path.append("..")
from qiskit.qasm import Qasm
import qiskit.unroll as unroll
import qiskit.localize as localize


def make_unrolled_circuit(fname, basis):
    """
    Create a graph representation of the QASM circuit.

    basis is a comma separated list of operation names.
    The circuit is unrolled to gates in the basis.
    """
    ast = Qasm(filename=fname).parse()
    u = unroll.Unroller(ast, unroll.CircuitBackend(basis.split(",")))
    u.execute()
    return u.be.C


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
coupling = localize.Coupling(couplingstr)
print("coupling = \n%s" % coupling)

# Third, do the mapping
c_prime = localize.swap_mapper(c, coupling)
print("c_prime.qasm() = \n%s" % c_prime.qasm(qeflag=True))
