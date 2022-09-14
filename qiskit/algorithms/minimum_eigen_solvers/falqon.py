from qiskit.opflow import StateFn, CircuitSampler, PauliExpectation, PauliSumOp, H
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate

from qiskit.quantum_info import SparsePauliOp

class FALQON:
    def __init__(self, cost_h, comm_h, driver_h=None):
        self.cost_h = cost_h
        self.comm_h = comm_h
        self.n_qubits = cost_h.num_qubits
        if driver_h is None:
            self.driver_h = PauliSumOp(
                SparsePauliOp.from_sparse_list([("X", [i], 1) for i in range(self.n_qubits)], num_qubits=self.n_qubits))
        else:
            self.driver_h = driver_h

    def build_maxclique_ansatz(self, delta_t, betas):
        circ = (H ^ self.cost_h.num_qubits).to_circuit()
        params = ParameterVector("beta", length=len(betas))
        for param in params:
            circ.append(PauliEvolutionGate(self.cost_h, time=delta_t), circ.qubits)
            circ.append(PauliEvolutionGate(self.driver_h, time=delta_t * param), circ.qubits)
        return circ

    def expval_circuit(self, ansatz, betas, measurement, sampler):
        params = dict(zip(ansatz.parameters, betas))
        composed = measurement.compose(StateFn(ansatz))
        exp = PauliExpectation().convert(composed)
        return sampler.convert(exp, params).eval()

    def run(self, n, backend, delta_t=0.03, beta_0=0.0, callback=None):
        comm_statefn = StateFn(self.comm_h).adjoint()
        cost_statefn = StateFn(self.cost_h).adjoint()

        betas = [beta_0]
        energies = []
        states = []

        sampler = CircuitSampler(backend=backend)
        for i in range(n):
            ansatz = self.build_maxclique_ansatz(delta_t, betas)
            beta = -1 * self.expval_circuit(ansatz, betas, comm_statefn, sampler)
            betas.append(beta)

            ansatz = self.build_maxclique_ansatz(delta_t, betas)
            energy = self.expval_circuit(ansatz, betas, cost_statefn, sampler)
            energies.append(energy)

            state = StateFn(ansatz).bind_parameters(dict(zip(ansatz.parameters, betas))).eval()
            states.append(state.primitive.data)

            if callback is not None:
                callback([betas, energies, states])

        return energies, betas, states[-1]