"""
Circuit compose example.

Author: Andrew Cross
"""
import sys
sys.path.append("..")
from qiskit.qasm import Qasm
import qiskit.unroll as unroll
import networkx as nx


def build_circuit(fname):
    """Build circuit."""
    ast = Qasm(filename=fname).parse()
    u = unroll.Unroller(ast, unroll.CircuitBackend(["u1", "u2", "u3", "cx"]))
    u.execute()
    return u.backend.circuit


if len(sys.argv) < 3:
    print("compose.py <file1> <file2> <\"back\"|\"front\"> [a:b] [a:b] ...\n")
    print("  file1 = base circuit")
    print("  file2 = circuit to compose")
    print("  \"back\"|\"front\" = how to compose file2")
    print("  [a:b] ... = wire map from file2 to file1 as \"q,0:r,1\" format")
    sys.exit(1)

c1 = build_circuit(sys.argv[1])
c2 = build_circuit(sys.argv[2])

wire_map = {}
for j in sys.argv[4:]:
    s = j.split(':')
    ql = s[0].split(',')
    qr = s[1].split(',')
    wire_map[(ql[0], int(ql[1]))] = (qr[0], int(qr[1]))

print("got wire_map = %s" % wire_map)

if sys.argv[3] == "front":
    c1.compose_front(c2, wire_map)
else:
    c1.compose_back(c2, wire_map)

print("composed circuits:\n%s" % c1.qasm())

print("View out.gml in a graph viewer such as Gephi")
nx.write_gml(c1.multi_graph, "out.gml", stringizer=str)

# print("c1.G.nodes(data=True) = \n%s\n" % c1.G.nodes(data=True))
# print("c1.G.edges(data=True) = \n%s\n" % c1.G.edges(data=True))
