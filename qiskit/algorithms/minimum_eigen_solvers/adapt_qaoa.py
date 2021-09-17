import numpy as np
from itertools import combinations_with_replacement, permutations, product

from typing import Optional, List, Callable, Union, Dict
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.opflow import OperatorBase, ExpectationBase
from qiskit.quantum_info import Operator
from qiskit.opflow.gradients import GradientBase
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit import QuantumCircuit
from qiskit import Aer, QuantumCircuit
from qiskit.opflow.primitive_ops import MatrixOp
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
from qiskit.algorithms.optimizers import Optimizer, SLSQP
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.algorithms import QAOA

class AdaptQAOA(QAOA):
    def __init__(
        self,
        optimizer: Optimizer = None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        mixer_list: list = None,
        initial_point: Optional[np.ndarray] = None,
        threshold: Optional[Callable[[int, float], None]] = None,
        gradient: Optional[Union[GradientBase, Callable[[Union[np.ndarray, List]],
                                                        List]]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[
            Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:

        super().__init__(
            optimizer=optimizer,
            reps=reps,
            initial_state=initial_state,
            initial_point=initial_point,
            gradient=gradient,
            expectation=expectation,
            include_custom=include_custom,
            max_evals_grouped=max_evals_grouped,
            callback=callback,
            quantum_instance=quantum_instance)
        self.initial_point =initial_point
        self.threshold = threshold
        self.mixer_list = mixer_list
        self.optimal_mixer_pool = [] # --------------> Will be appending optimal mixers to this.

    def _check_operator_ansatz(self, operator: OperatorBase) -> OperatorBase:
        # Recreates a circuit based on operator parameter.
        if operator.num_qubits != self.ansatz.num_qubits:
            self.ansatz = QAOAAnsatz(
                operator, self._reps, initial_state=self._initial_state, 
                mixer_operator= (self.optimal_mixer_pool if self.optimal_mixer_pool else self.mixer_list)
            ).decompose()  # TODO remove decompose once #6674 is fixed

        
    def compute_energy_gradient(self, previous_state=None):
        from qiskit.opflow import commutator
        """Computes the energy gradient of the cost operator wrt the mixer poolat an ansatz layer specified by 
            the input 'state' and initial point.
            Returns: The mixer operator with the largest energy gradient along with the associated energy
                     gradient."""

        if not isinstance(self.cost_operator, MatrixOp):
            cost_op = MatrixOp(Operator(self.cost_operator))

        if not previous_state:
            previous_state = self.initial_state
        
        n_pts = len(self.initial_point)
        gamma, beta = self.initial_point[:int(n_pts/2)], self.initial_point[int(n_pts/2):] 
        "Might be useful to set these ^^^^ as private class variables (self.gamma, self.beta) ?"

        exp_Hc = (gamma*cost_op).exp_i()
        exp_Hc, exp_Hc_ad = [_.to_matrix() for _ in [exp_Hc, exp_Hc.adjoint()]] 

        mixer_gradient = []
        for mixer in self.mixer_list:
            energy_grad_op = exp_Hc_ad@(commutator(1j*cost_op,mixer).to_matrix())@exp_Hc
            energy_gradient = np.abs(self.find_minimum(MatrixOp(Operator(energy_grad_op))).eigenvalue)
            mixer_gradient.append(energy_gradient)
        return self.mixer_list[np.argmax(mixer_gradient)], np.max(energy_gradient)

    def constuct_adapt_ansatz(self, operator: OperatorBase) -> OperatorBase:
        energy_norm, p = 0, 0
        while energy_norm > self.threshold and p<self.reps:
            pass
        """"
            Loop goes here, steps are roughly as follows:
            ---------------------------------------------
        (0) Outside of the loop:
            - Initialise p = 0 (max reps counter) and call adapt_mixer_pool to construct mixer
              pool. Would be good to place adapt_mixer_pool in adapt_class setter in the case
              when mixer_list = None.
            - Initialise a new variable, previous_state = self.initial state.
              This is passed into compute_energy_gradient and will be updated every loop i
              teration.
        
        (1) With previous_state = self.initial_state, call compute_energy_gradient
            to calculate the best mixer & associated energy gradient. 

            (1.5) Doing this requires self.cost_operator to be pre-set but I'm not sure on the
                best way to do this. 
                    - My current approach is to simply pass it as an argument to this
                    function.
                  
        (2) Append the optimal mixer computed in (1) to self.optimal_mixer_pool. Then call:

            - trial_ansatz = QAOAAnsatz(reps=p, initial_state=previous_state).construct_circuit(operator)

        (3) Next, optimize trial_ansatz & compute the minimum cost via,
            - cost_optimal_param_dict = self.find_minimum(ansatz=trial_ansatz)
            - set cost_optimal_param_dict['cost'] (or w/e key == min. cost) = energy_norm
        (4) Update initialised parameters,
            - previous_state = self.get_optimal_circuit() -----> might not be a call to self idk
            - p += 1
        (5) Repeat 1-4.
        """""


    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """
        Returns:
            Returns the initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """
        Args:
            initial_state: Initial state to set.
        """
        self._initial_state = initial_state

    @property
    def mixer(self) -> Union[QuantumCircuit, OperatorBase]:
        """
        Returns:
            Returns the mixer.
        """
        return self._mixer

    @mixer.setter
    def mixer(self, mixer: Union[QuantumCircuit, OperatorBase]) -> None:
        """
        Args:
            mixer: Mixer to set.
        """
        self._mixer = mixer




def adapt_mixer_pool(num_qubits: int, add_single: bool = True, add_multi: bool = True) -> List:
    """
    Gets all combinations of mixers in desired set (standard qaoa mixer, single qubit mixers, multi qubit mixers)
    Args:
        num_qubits: number of qubits
        add_single: whether to add single qubit to mixer pool (not standard qaoa x mixers)
        add_multi: whether to add multi qubit to mixer pool

    Returns:
        List of all possible combinations of mixers.
    """
    # always include the all x's:
    mixer_pool = ["X" * num_qubits]
    if add_single:
        # y's
        mixer_pool.append("Y" * num_qubits)
        mixer_pool += [i * "I" + 'X' + (num_qubits - i - 1)
                       * "I" for i in range(num_qubits)]
        mixer_pool += [i * "I" + 'Y' + (num_qubits - i-1)
                       * "I" for i in range(num_qubits)]
    if add_multi:
        indicies = list(permutations(range(num_qubits), 2))
        indicies = list(set(tuple(sorted(x)) for x in indicies))
        combos = list(combinations_with_replacement(['X', 'Y', 'Z'], 2))
        full_multi = list(product(indicies, combos))
        for item in full_multi:
            iden_str = list("I" * num_qubits)
            iden_str[item[0][0]] = item[1][0]
            iden_str[item[0][1]] = item[1][1]
            mixer_pool.append(''.join(iden_str))
    return mixer_pool

# num, reps = 6, 2
# mixers_list = adapt_mixer_pool(num)
# mixerop_pool = []
# for mix_str in mixers_list:
#     qr = QuantumRegister(num)
#     qc = QuantumCircuit(qr)
#     for i, mix in enumerate(mix_str):
#         qiskit_dict = {"I": IGate(), "X": XGate(), "Y":YGate(), "Z":ZGate()}

#         mix_qis_gate = qiskit_dict[mix]
#         qc.append(mix_qis_gate, [i])
#     mixerop_pool.append(MatrixOp(Operator(qc)))
# G = nx.random_regular_graph(5, 6, seed=1234) # connectivity, vertices
# for (u, v) in G.edges():
#     G.edges[u,v]['weight'] = random.randint(0,1000)/1000

# Hc = MatrixOp(build_maxcut_hamiltonian(G).full())
# init_params_gamma  = [0.01 for i in range(0, reps)] 
# init_params_beta = [-np.pi/4 for i in range(0, reps)]
# init_guess = init_params_gamma+init_params_beta


# adqaoa = AdaptQAOA(optimizer=SLSQP(), reps=reps, mixer_list=mixerop_pool, quantum_instance=Aer.get_backend('statevector_simulator'),
#             initial_point=init_guess)
# # print(adqaoa.compute_minimum_eigenvalue(Hc))
# print(adqaoa.construct_circuit(init_guess,Hc))
