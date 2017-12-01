import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import StageBase, StageInputOutput, StageError
import qiskit.mapper as mapper

class SwapMapperStage(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'SwapMapperStage'

    def handle_request(self, input):
        dag_circuit = input.get('dag_circuit')
        coupling = input.get('coupling')
        final_layout = input.get('layout')
        compiled_dag_circuit, final_layout = mapper.swap_mapper(
            dag_circuit, coupling, layout, trials=20)

        input.insert('dag_circuit', compiled_dag_circuit)
        input.insert('layout', final_layout)
        input.insert('qasm_circuit', compiled_dag_circuit.qasm())
        input.insert('coupling', coupling)

        return input

    def check_precondition(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists(['dag_circuit','coupling', 'layout']):
            return False

        return True