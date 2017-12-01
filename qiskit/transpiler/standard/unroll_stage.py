import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import StageBase, StageInputOutput, StageError
import qiskit.qasm as qasm
import qiskit.unroll as unroll

class UnrollStage(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'UnrollStage'

    def handle_request(self, input):
        backend_target = input.get('unroll_backend_target')
        qasm_circuit = input.get('qasm_circuit')

        try:
            basis_gates = input.get('basis_gates')
        except StageError:
            # TODO Is this a safe default for every use-case?
            basis_gates = 'u1,u2,u3,cx,id'

        if backend_target == 'dag':
            backend = unroll.DAGBackend
            circuit_key = 'dag_circuit'
        elif backend_target == 'json':
            backend = unroll.JsonBackend
            circuit_key = 'json_circuit'
        elif backend_target == 'circuit':
            circuit_key = 'circuit_circuit'
            backend = unroll.CircuitBackend
        elif backend_target == 'printer':
            circuit_key = 'printer_circuit'
            backend = unroll.PrinterBackend

        ast = qasm.Qasm(data=qasm_circuit).parse()
        unrolled = unroll.Unroller(ast, backend(basis_gates.split(',')))
        circuit_unrolled = unrolled.execute()

        input.insert(circuit_key, circuit_unrolled)
        # We want insert basis_gates for future unrolling
        input.insert('basis_gates', basis_gates)

        return input

    def check_precondition(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists('unroll_backend_target', 'qasm_circuit'):
            return False

        return True


