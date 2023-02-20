from collections.abc import Sequence

from qiskit.primitives import BaseEstimator

import numpy as np

from .minimum_eigensolver import MinimumEigensolver, MinimumEigensolverResult
from ..list_or_dict import ListOrDict
from ...opflow import PauliSumOp, H, commutator
from ...quantum_info import SparsePauliOp
from ...quantum_info.operators.base_operator import BaseOperator
from ...circuit import ParameterVector
from ...circuit.library import PauliEvolutionGate


class FALQON(MinimumEigensolver):

    def __init__(self,
                 estimator: BaseEstimator,
                 n: int = 1,
                 delta_t: float = 0.3,
                 initial_point: Sequence[float] = None
                 ):
        super().__init__()

        self.estimator = estimator
        self.n = n
        self.delta_t = delta_t
        self.initial_point = initial_point

    def build_ansatz(self, operator, driver_h, betas):
        circ = (H ^ operator.num_qubits).to_circuit()
        params = ParameterVector("beta", length=len(betas))
        for param in params:
            circ.append(PauliEvolutionGate(operator, time=self.delta_t), circ.qubits)
            circ.append(PauliEvolutionGate(driver_h, time=self.delta_t * param), circ.qubits)
        return circ

    def compute_minimum_eigenvalue(self,
                                   operator: BaseOperator | PauliSumOp,
                                   aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None
                                   ) -> "MinimumEigensolverResult":

        n_qubits = operator.num_qubits
        driver_h = PauliSumOp(SparsePauliOp.from_sparse_list([("X", [i], 1) for i in range(n_qubits)],
                                                             num_qubits=n_qubits))
        comm_h = (complex(0, 1) * commutator(driver_h, operator)).reduce()

        if self.initial_point is None:
            betas = [0.0]
        else:
            betas = self.initial_point

        energies = []

        for i in range(self.n):
            ansatz = self.build_ansatz(operator, driver_h, betas)
            beta = -1 * self.estimator.run(ansatz, comm_h, betas).result().values[0]
            betas.append(beta)

            ansatz = self.build_ansatz(operator, driver_h, betas)
            energy = self.estimator.run(ansatz, operator, betas).result().values[0]
            energies.append(energy)

        return self._build_falqon_result(energies)

    def _build_falqon_result(self, energies):
        result = MinimumEigensolverResult()
        result.aux_operators_evaluated = None
        result.eigenvalue = np.min(np.asarray(energies))
        return result
