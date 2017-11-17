from qiskit.transpiler import (StageBase, StageInputOutput, StageError)

""" This is the main transpiler pipeline manager class. Here we will register
the stages and build the pipeline. """
class Pipeline(object):
    def __init__(self):
        self.input_output = StageInputOutput()
        self.pipeline = []

    def register_stage(self, stage):
        if not isinstance(stage, StageBase):
            raise StageError('Error: Stages must inherit from StageBase')

        self.pipeline.append(stage)

    def execute(self, init_input_values = {}):
        self._init_input(init_input_values)
        # TODO: Check that the pipeline has at least one
        for stage in self.pipeline:
            try:
                self.input_output = stage.handle_request(self.input_output)
            except StageError as ex:
                raise QISKitError(str(ex)) from ex

        # TODO: Warn if the result is None?. The reasoning behind this, is that
        # a user could forgot to set the final result in the Stages... but,
        # 'None' could be considered a valid result.
        return self.input_output.result

    def _init_input(self, values):
        if not isinstance(values, dict):
            raise PipelineError('Initial input value must be a dictionary!')

        for key, val in values.items():
            self.input_output.insert(key, val)
