import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import StageBase, StageInputOutput, StageError

class ReturnJsonStage(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'TransformStage'

    def handle_request(self, input):
        json_circuit = input.get('json_circuit')
        input.remove('json_circuit')
        input.result = json_circuit
        return input

    def check_precondition(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists(['dag_circuit', 'json_circuit']):
            return False

        self.format = input.get('format')
        if self.format != 'json':
            return False

        return True