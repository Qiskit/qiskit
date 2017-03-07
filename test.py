import sys
import qx_sdk.Qasm as Qasm
import qx_sdk.QasmBackends as QasmBackends
import qx_sdk.QasmInterpreters as QasmInterpreters
import networkx as nx

if len(sys.argv) < 2:
  print("test.py <file> [backend]\n")
  print("backends:")
  print("  circuit - CircuitBackend with basis u1, u2, u3, cx")
  print("  base    - BaseBackend") 
  sys.exit(1)

# Parse the QASM file
try:
  q = Qasm.Qasm(filename=sys.argv[1])
  ast = q.parse()

except Qasm.qasm.QasmException as e:
  print(e.msg)

except Exception as e:
  print(sys.exc_info()[0], 'Exception parsing qasm file')
  traceback.print_exc()

print("AST")
print("-----------------------------------------")
ast.to_string(0)
print("-----------------------------------------")
print("QASM source output from the AST")
print("-----------------------------------------")
print(ast.qasm())
print("-----------------------------------------")

select = "base"
if len(sys.argv) >= 3:
  select = sys.argv[2]

# Create Backend
if select == "circuit":
  be = QasmBackends.CircuitBackend()
elif select == "base":
  be = QasmBackends.BaseBackend()
else:
  print("unknown backend \"%s\""%sys.argv[2])
  sys.exit(1)

be.set_basis(["u1","u2","u3","cx"])

# Interpret QASM to backend
unroller = QasmInterpreters.Unroller(ast)
unroller.set_backend(be)
unroller.execute()

# Display results for circuit backend
if select == "circuit":
  print("QASM source output from the circuit graph")
  print("-----------------------------------------")
  print(be.cg.qasm())
  print("-----------------------------------------")

  print("size    = %d"%be.cg.size())
  print("depth   = %d"%be.cg.depth())
  print("width   = %d"%be.cg.width())
  print("bits    = %d"%be.cg.num_cbits())
  print("factors = %d"%be.cg.num_tensor_factors())

  print("View out.gml in a graph viewer such as Gephi")
  nx.write_gml(be.cg.G,"out.gml",stringizer=str)
