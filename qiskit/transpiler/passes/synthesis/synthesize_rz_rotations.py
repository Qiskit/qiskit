from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit._accelerate.synthesize_rz_rotations import synthesize_rz_rotations


class SynthesizeRZRotations(TransformationPass):
    """Replace RZ gates with Clifford+T decompositions using gridsynth.

    This pass replaces all single-qubit RZ rotation gates with sequences
    of Clifford+T gates (H, S, T gates) using the gridsynth algorithm.

    Args:
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
