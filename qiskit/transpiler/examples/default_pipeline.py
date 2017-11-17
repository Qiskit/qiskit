import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from qiskit.transpiler import Pipeline, StageBase, StageInputOutput, StageError

# TODO: Remove StageInputOutput depdendency from Stages?. Pipeline would manage it
class ShowListStage(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'ShowList'

    def handle_request(self, input):
        if not self._check_preconditions(input):
            return input

        list = input.get('list')
        for elem in list:
            print('ShowListStage: elem = {}'.format(elem))

        return input

    def _check_preconditions(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists('list'):
            return False
        return True


class AddElementStage(StageBase):
    def __init__(self):
        self.name = 'AddElement'

    def get_name(self, name):
        return self.name

    def handle_request(self, input):
        if not self._check_preconditions(input):
            return input

        list = input.get('list')
        list.append(5)
        print('AddElementStage: Add a new element 5 to the list')
        return input

    def _check_preconditions(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists('list'):
            return False
        return True

class ChangeElementStage(StageBase):
    def __init__(self):
        self.element_to_change = None

    def get_name(self, name):
        return 'ChangeElement'

    def handle_request(self, input):
        if not self._check_preconditions(input):
            return input

        self.element_to_change += self.element_to_change
        list.append(self.element_to_change)
        print('ChangeElementStage: New element value {} changed in the list'
              .format(self.element_to_change))
        return input

    def _check_preconditions(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        try:
            self.element_to_change = input.get('list')[-1:]
        except:
            return False

        import numbers
        if not isinstance(self.element_to_change, numbers.Number):
            return False

        return True

class SummationStage(StageBase):
    def __init__(self):
        self._list = None

    def get_name(self, name):
        return 'Summation'

    def handle_request(self, input):
        if not self._check_preconditions(input):
            return input

        summation = sum(self._list)

        print('SummationStage: The summation of all the list elements is: {}'
              .format(summation))
        
        input.result = summation
        return input

    def _check_preconditions(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        try:
            self._list = input.get('list')
        except:
            return False

        if not isinstance(self._list, list):
            return False

        return True



if __name__ == '__main__':
    test_pipeline = Pipeline()
    test_pipeline.register_stage(AddElementStage())
    test_pipeline.register_stage(ChangeElementStage())
    test_pipeline.register_stage(SummationStage())
    test_pipeline.register_stage(ShowListStage())

    result = test_pipeline.execute(init_input_values = {'list': [1,2,3,4]})

    print('Pipeline result: {}'.format(result))

