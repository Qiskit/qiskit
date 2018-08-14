from qiskit import qasm, unroll


def buildAST(qasm_file):
    if not qasm_file:
        print('"Not filename provided')
        return {"status": "Error", "result": "Not filename provided"}
    ast = qasm.Qasm(filename=qasm_file).parse()  # Node (AST)

    return ast


def buildCircuit(ast, basis_gates=None):
    unrolled_circuit = unroll.Unroller(ast=ast, backend=unroll.DAGBackend(basis_gates)) #CircuitBackend
    unrolled_circuit.execute()
    circuit_unrolled = unrolled_circuit.backend.circuit  # circuit DAG
    return circuit_unrolled
