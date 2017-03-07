import sys
import qx_sdk.Qasm as Qasm
import qx_sdk.QasmBackends as QasmBackends
import qx_sdk.QasmInterpreters as QasmInterpreters
import networkx as nx

def build_circuit(fname,basis):
  q = Qasm.Qasm(filename=fname)
  ast = q.parse()
  be = QasmBackends.CircuitBackend()
  be.set_basis(basis)
  u = QasmInterpreters.Unroller(ast,be)
  u.execute()
  return be.cg

if len(sys.argv) < 3:
  print("csub2.py <file1> <basis1> <file2> <basis2> <opname> <w1> <w2> ... <wn>\n")
  print("  file1 = main circuit")
  print("  basis1 = basis for main circuit, \"gate,gate,...\" format")
  print("  file2 = input circuit to substitute")
  print("  basis2 = basis for input circuit, \"gate,gate,...\" format")
  print("  opname = operation to replace")
  print("  w1 ... = wire order for opname, \"q,0\" format")
  print("")
  print("Generates a circuit for each opname in file1")
  sys.exit(1)

basis1 = sys.argv[2].split(",")
basis2 = sys.argv[4].split(",")

print("got basis1 = %s"%basis1)
print("got basis2 = %s"%basis2)

c1 = build_circuit(sys.argv[1],basis1)
c2 = build_circuit(sys.argv[3],basis2)

opname = sys.argv[5]

wires = []
for j in sys.argv[6:]:
  q = j.split(',')
  wires.append((q[0],int(q[1])))

print("got wires = %s"%wires)

nlist = c1.get_named_nodes(opname)

print("%d operations in c1 named %s"%(len(nlist),opname))

for i in range(len(nlist)):

  n = nlist[i]
  nd = c1.G.node[n]
  # ignoring nd["condition"]
  print("%d -- %s %s %s %s %s"%(i,nd["name"],nd["type"],nd["qargs"],\
                             nd["cargs"],nd["params"]))
  c1p = c1.clone()
  c1p.remove_descendants_of(n)
  c1p.substitute_circuit_one(n,c2,wires)
  print(c1p.qasm())
