from qiskit.transpiler._basepasses import AnalysisPass

class GateAnalysis(AnalysisPass):

    def __init__(self):
        super().__init__()

    def run(self, dag):

        self.property_set['gate_count'] = dag.multi_graph.number_of_nodes()
