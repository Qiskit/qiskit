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
        if not self._check_preconditions(input):
            return input

        backend_target = input.get('backend_target')
        input.remove('unroll_target_backend')
        
        try:
            basis_gates = input.get('basis_gates')
        except StageError:
            # TODO Is this a safe default for every use-case?
            basis_gates = 'u1,u2,u3,cx,id'

        if backend_target == 'dag':
            backend = unroll.DAGBackend
        elif backend_target == 'json':
            backend = unroll.JsonBackend
        elif backend_target == 'circuit'
            backend = unroll.CircuitBackend
        elif backend_target == 'printer'
            backend = unroll.PrinterBackend

        ast = qasm.Qasm(data=qasm_circuit).parse()
        unrolled = unroll.Unroller(ast, backend(basis_gates.split(',')))
        dag_circuit_unrolled = unrolled.execute()

        input.insert('dag_circuit', dag_circuit_unrolled)

        return input

    def _check_preconditions(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists('unroll_target_backend', 'qasm_circuit'):
            return False

        return True


