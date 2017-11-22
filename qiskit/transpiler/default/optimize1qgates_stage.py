import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import StageBase, StageInputOutput, StageError
import qiskit.mapper as mapper

class Optimize1qGatesState(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'CoupOptimize1qGatesState'

    def handle_request(self, input):
        if not self._check_preconditions(input):
            return input

        dag_circuit = input.get('dag_circuit')

        input.insert('dag_circuit',
            mapper.optimize_1q_gates(dag_circuit))

        return input

    def _check_preconditions(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists('dag_circuit'):
            return False

        return True