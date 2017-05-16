"""
Circuit substitution example, substituting one at a time.

Substitute the input circuit file2 in place of each occurence of the
operation opname in the main circuit file1, one at a time, generating output
QASM each time. Each file is unrolled to the given basis and the mapping
from input wires of opname to wires of the main circuit is given by the wire
order w1, w2, ..., wn.

Author: Andrew Cross
        Jesus Perez <jesusper@us.ibm.com>
"""

import sys
import os

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from qiskit.qasm import Qasm
import qiskit.unroll as unroll


def build_circuit(fname, basis):
    """Return a circuit given a QASM file."""
    ast = Qasm(filename=fname).parse()
    u = unroll.Unroller(ast, unroll.CircuitBackend(basis))
    u.execute()
    return u.backend.circuit


if len(sys.argv) < 3:
    print("substituteone.py <file1> <basis1> <file2> <basis2> ", end="")
    print("<opname> <w1> <w2> ... <wn>\n")
    print("  file1 = main circuit")
    print("  basis1 = basis for main circuit, \"gate,gate,...\" format")
    print("  file2 = input circuit to substitute")
    print("  basis2 = basis for input circuit, \"gate,gate,...\" format")
    print("  opname = operation to replace")
    print("  w1 ... = wire order for opname, \"q,0\" format\n")
    print("Generates a circuit for each opname in file1")
    sys.exit(1)

basis1 = sys.argv[2].split(",")
basis2 = sys.argv[4].split(",")
print("got basis1 = %s" % basis1)
print("got basis2 = %s" % basis2)

c1 = build_circuit(sys.argv[1], basis1)
c2 = build_circuit(sys.argv[3], basis2)

opname = sys.argv[5]

wires = []
for j in sys.argv[6:]:
    q = j.split(',')
    wires.append((q[0], int(q[1])))
print("got wires = %s" % wires)

nlist = c1.get_named_nodes(opname)
print("%d operations in c1 named %s" % (len(nlist), opname))

for i in range(len(nlist)):
    n = nlist[i]
    nd = c1.multi_graph.node[n]
    # ignoring nd["condition"]
    print("%d -- %s %s %s %s %s" % (i, nd["name"], nd["type"], nd["qargs"],
                                    nd["cargs"], nd["params"]))
    c1p = c1.deepcopy()
    c1p.remove_descendants_of(n)
    c1p.substitute_circuit_one(n, c2, wires)
    print(c1p.qasm())
