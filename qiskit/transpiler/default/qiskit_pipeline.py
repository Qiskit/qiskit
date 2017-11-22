import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import Pipeline, StageBase, StageError

from . import UnrollStage, CouplingStage, SwapMapperStage, DirectionMapperStage,
              CxCancellationStage, Optimize1qGatesState, TransformStage

class QiskitPipeline():
    def __init__(self):
        self.pipeline = Pipeline()
        self.pipeline.register_stage(UnrollStage)
        self.pipeline.register_stage(CouplingStage)
        self.pipeline.register_stage(SwapMapperStage)
        self.pipeline.register_stage(UnrollStage)
        self.pipeline.register_stage(UnrollStage) # This should be skipped
        self.pipeline.register_stage(DirectionMapperStage)
        self.pipeline.register_stage(UnrollStage)
        self.pipeline.register_stage(CxCancellationStage)
        self.pipeline.register_stage(Optimize1qGatesState)
        self.pipeline.register_stage(UnrollStage)
        self.pipeline.register_stage(TransformStage)
        self.pipeline.register_stage(UnrollStage)

    def execute(self, qasm_circuit, basis_gates, coupling_map, layout,
                get_layout, format):
        try:
            result = self.pipeline.execute({'qasm_circuit': qasm_circuit,
                                            'basis_gates': basis_gates,
                                            'coupling_map': coupling_map,
                                            'layout': layout,
                                            'get_layout', get_layout,
                                            'format', format})
        except StageError as ex:
            raise QISKitError(str(ex)) from ex

        return result