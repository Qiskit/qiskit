"""
Unroll to a circuit and display results.

Author: Andrew Cross
"""
import sys
sys.path.append("..")
from qiskit.qasm import Qasm
import qiskit.unroll as unroll
import networkx as nx


if len(sys.argv) < 2:
    print("circuitunroll.py <qasmfile>\n")
    print("basis = u1, u2, u3, cx")
    sys.exit(1)

ast = Qasm(filename=sys.argv[1]).parse()

print("AST")
print("-----------------------------------------")
ast.to_string(0)
print("-----------------------------------------")
print("QASM source output from the AST")
print("-----------------------------------------")
print(ast.qasm())
print("-----------------------------------------")

basis = ["u1", "u2", "u3", "cx"]
unroller = unroll.Unroller(ast, unroll.CircuitBackend(basis))
unroller.execute()
C = unroller.backend.circuit

print("QASM source output from the circuit graph")
print("-----------------------------------------")
print(C.qasm())
print("-----------------------------------------")

print("size    = %d" % C.size())
print("depth   = %d" % C.depth())
print("width   = %d" % C.width())
print("bits    = %d" % C.num_cbits())
print("factors = %d" % C.num_tensor_factors())

print("View out.gml in a graph viewer such as Gephi")
nx.write_gml(C.multi_graph, "out.gml", stringizer=str)
