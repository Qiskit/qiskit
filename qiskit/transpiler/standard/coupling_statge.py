import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import StageBase, StageInputOutput, StageError
import qiskit.mapper as mapper

class CouplingStage(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'CouplingStage'

    def handle_request(self, input):
        coupling_map = input.get('coupling_map')
        input.insert('coupling', mapper.Coupling(coupling_map))

        return input

    def check_precondition(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists('coupling_map'):
            return False

        return True
