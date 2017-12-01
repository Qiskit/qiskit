import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import StageBase, StageInputOutput, StageError
import qiskit.mapper as mapper

class DirectionMapperStage(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'DirectionMapper'

    def handle_request(self, input):
        coupling = input.get('coupling')
        dag_circuit = input.get('dag_circuit')

        dag_output =  mapper.direction_mapper(dag_circuit, coupling)

        input.insert('dag_circuit', dag_output)
        input.insert('qasm_circuit', dag_circuit.qasm())
        input.insert('unroller_backend_target', 'dag')
        return input

    def check_precondition(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists(['coupling','dag_circuit']):
            return False

        return True