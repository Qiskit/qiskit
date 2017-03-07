import sys
import qx_sdk.Qasm as Qasm
import qx_sdk.QasmBackends as QasmBackends
import qx_sdk.QasmInterpreters as QasmInterpreters

if len(sys.argv) < 2:
  print("unroll.py <file> [basis]\n")
  print("  file   - input QASM file")
  print("  [basis] - optional basis, \"gate,gate,...\" format")
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

if ast is None:
  sys.exit(1)

if len(sys.argv) > 2:
  basis = sys.argv[2].split(",")
else:
  basis = []

be = QasmBackends.BaseBackend()
be.set_basis(basis)
unroller = QasmInterpreters.Unroller(ast)
unroller.set_backend(be)
unroller.execute()
