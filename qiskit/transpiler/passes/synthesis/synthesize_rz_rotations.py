from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.synthesize_rz_rotations import synthesize_rz_rotations


class SynthesizeRZRotations(TransformationPass):
    """Replace RZ gates with Clifford+T decompositions in an efficient manner.

    This pass replaces all single-qubit RZ rotation gates with sequences
    of Clifford+T gates. We first canonicalize angles based on the 4π cyclicity 
    of RZ gates, and further utilize the property RZ(θ+2π) = e^{iπ} RZ(θ) to map
    angles to the [0, 2π) range to limit the number of distinct angles synthesized.
    We then iterate over the dag to identify RZ gates and replace them with their
    Clifford+T approximations.

    Args:
        [Change this to approximation degree in later iteration to maintain
        consistency across other passes]
        
        epsilon: Precision parameter for gridsynth approximation.
        Default: 1e-10.

        [need to elaborate more, WIP]
    """

    def __init__(self, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on a DAG.

        Args:
            dag: the input DAG.

        Returns:
            The output DAG.
        """
        new_dag = synthesize_rz_rotations(dag, self.epsilon)

        if new_dag is None:
            return dag

        return new_dag
