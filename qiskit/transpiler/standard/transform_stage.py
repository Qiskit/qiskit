import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import StageBase, StageInputOutput, StageError
import qiskit.qasm as qasm
import qiskit.unroll as unroll

class TransformStage(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'TransformStage'

    def handle_request(self, input):
        output_format = input.get('format')
        dag_circuit = input.get('dag_circuit')

        if output_format == 'dag':
            transformed_circuit = dag_circuit
        elif output_format == 'json':
            input.insert('unroll_backend_target', 'json')
            input.insert('qasm_circuit', dag_circuit.qasm())
        elif output_format == 'qasm':
            transformed_circuit = dag_circuit.qasm()
        else:
            raise StageError('Unrecognized circuit format: {}'.format(
                    output_format))

        input.result = transformed_circuit
        return input

    def check_precondition(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists(['dag_circuit','format'])
            return False

        return True