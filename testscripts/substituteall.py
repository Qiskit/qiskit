"""
Circuit substitution example, substituting all at once.

Substitute the input circuit file2 in place of every occurence of the
operation opname in the main circuit file1. Each file is unrolled to the
given basis and the mapping from input wires of opname to wires of the
main circuit is given by the wire order w1, w2, ..., wn.

Author: Andrew Cross
"""
import sys
sys.path.append("..")
from qiskit.qasm import Qasm
import qiskit.unroll as unroll
import networkx as nx


def build_circuit(fname, basis):
    """Return a circuit given a QASM file."""
    ast = Qasm(filename=fname).parse()
    u = unroll.Unroller(ast, unroll.CircuitBackend(basis))
    u.execute()
    return u.backend.circuit


if len(sys.argv) < 3:
    print("substituteall.py <file1> <basis1> <file2> <basis2> ", end="")
    print("<opname> <w1> <w2> ... <wn>\n")
    print("  file1 = main circuit")
    print("  basis1 = basis for main circuit, \"gate,gate,...\" format")
    print("  file2 = input circuit to substitute")
    print("  basis2 = basis for input circuit, \"gate,gate,...\" format")
    print("  opname = operation to replace")
    print("  w1 ... = wire order for opname, \"q,0\" format\n")
    print("Replaces all instances of opname in file1")
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

# print("c1.G.nodes(data=True) = \n%s\n" % c1.G.nodes(data=True))
# print("c1.G.edges(data=True) = \n%s\n" % c1.G.edges(data=True))

c1.substitute_circuit_all(opname, c2, wires)

# print("c1.G.nodes(data=True) = \n%s\n" % c1.G.nodes(data=True))
# print("c1.G.edges(data=True) = \n%s\n" % c1.G.edges(data=True))

print("View out.gml in a graph viewer such as Gephi")
nx.write_gml(c1.multi_graph, "out.gml", stringizer=str)

print("circuit after substitution:\n%s" % c1.qasm())
