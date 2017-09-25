#!/usr/bin/env python
try:
    import qiskit
except ImportError as ierr:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    import qiskit
from qiskit import QuantumProgram
from qiskit.simulators._qasmsimulator import QasmSimulator
import qiskit.qasm as qasm
import qiskit.unroll as unroll
if __name__ == '__main__':
    from _random_qasm_generator import RandomQasmGenerator
else:
    from test.python._random_qasm_generator import RandomQasmGenerator


def addPrefix(prefix, circuitNameList):
    answer = []
    for each in circuitNameList:
        answer.append(prefix+"_"+each)
    return answer


if __name__ == "__main__":
    #seed = 0
    #nQbits = 20
    #depth = 100
    nCircuits = 1

    for nQbits in range(26,31):
        for depth in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            for seed in range(nCircuits):
                qasmGen = RandomQasmGenerator(seed=seed, maxQubits=nQbits, minQubits=nQbits,
                                              maxDepth=depth, minDepth=depth)
                qasmGen.add_circuits(1)
                circName = addPrefix("qbits"+str(nQbits)+"_depth"+str(depth), qasmGen.get_circuit_names())
                origName = qasmGen.get_circuit_names()
                qasm = qasmGen.getProgram().get_qasm(origName[0])
                outfile = open(circName[0]+".qasm", "w")
                outfile.write(qasm)
                outfile.close()
                print("Writing to "+circName[0])
