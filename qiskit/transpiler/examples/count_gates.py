import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from qiskit.transpiler import StageBase
from qiskit import unroll, qasm

class CountGates(StageBase):
    def __init__(self):
        self.count = {}

    def get_name(self):
        return 'CountGates'

    def handle_request(self):
        pass

    def check_preconditions(self):
        self.depends_on(['unroll'])

    def run(self, dag):
        for node in dag.multi_graph.nodes(data=True):
            if not type(node[1]['name']) is str: continue
            if node[1]['name'] is 'U':
                name = "%s%i" % (node[1]['name'],len(node[1]['qargs']))
            else:
                name = node[1]['name']
            self.count.setdefault(name, 0)
            self.count[name] += 1
        return dag

qasm_code = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
// -X-.-----
// -Y-+-S-.-
// -Z-.-T-+-
// ---+-H---
x q[0];
y q[1];
z q[2];
cx q[0], q[1];
cx q[2], q[3];
s q[1];
t q[2];
h q[3];
cx q[1], q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
'''

def main():
    ast = qasm.Qasm(data=qasm_code).parse()
    unrolled_circuit = unroll.Unroller(ast,unroll.DAGBackend())
    dag = unrolled_circuit.execute()

    count_gates_pass = CountGates()
    count_gates_pass.run(dag)

    for name,count in count_gates_pass.count.items():
        print('%s: %i' % (name,count))

if __name__ == "__main__":
    main()