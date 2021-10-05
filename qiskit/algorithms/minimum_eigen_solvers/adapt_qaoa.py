import numpy as np
from itertools import combinations_with_replacement, permutations, product

from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.expectations.expectation_factory import ExpectationFactory
from typing import Iterable, Optional, List, Callable, Union, Dict
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.opflow import OperatorBase, ExpectationBase
from qiskit.quantum_info import Operator
from qiskit.opflow.gradients import GradientBase
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.opflow.primitive_ops import MatrixOp
from qiskit.algorithms.optimizers import Optimizer
#CHANGEBACK TODO
from ...circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.algorithms import QAOA

def translate_mixer_str_opn(opn: str) -> Dict:
    opn = opn.lower()
    if opn == 'multi':
        return {"add_single": True, "add_multi":True}
    elif opn == 'singular':
        return {"add_single": True, "add_multi":False}
    elif opn == 'single':
        return {"add_single": False, "add_multi":False}
    else:
        #TODO raise errors
        return None

class AdaptQAOA(QAOA):
    def __init__(
        self,
        optimizer: Optimizer = None,
        max_reps: int = 5,
        initial_state: Optional[QuantumCircuit] = None,
        gamma_init: Optional[float] = 0.01,
        beta_init: Optional[float] = np.pi/4,
        mixer_pool: Optional[Union[str,List]] = None,
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
            reps=max_reps,
            initial_state=initial_state,
            gradient=gradient,
            expectation=expectation,
            include_custom=include_custom,
            max_evals_grouped=max_evals_grouped,
            callback=callback,
            quantum_instance=quantum_instance)
        self.gamma_init, self.beta_init = gamma_init, beta_init
        self.threshold = threshold
        self.mixer_pool = mixer_pool
        self.optimal_mixer_list = [] # --------------> Will be appending optimal mixers to this.
        self.cost_operator = None
        self.max_reps = max_reps
        self.best_gamma = []
        self.best_beta = []

        if self.mixer_pool is None:
            self.mixers = 'multi'
        self._check_mixer_pool()


        
    def _check_operator_ansatz(self, operator: OperatorBase, mixer_list = None) -> OperatorBase:
        # Recreates a circuit based on operator parameter.
        if mixer_list is None:
            mixer_list = self.optimal_mixer_list
        
        self.ansatz = QAOAAnsatz(
            operator, self._reps, initial_state = self._initial_state,
            mixer_operator = mixer_list
        ).decompose()  # TODO remove decompose once #6674 is fixed

        return self.ansatz

  
    def compute_energy_gradient(self, mixer,cost_operator, ansatz):
        from qiskit.opflow import commutator
        """Computes the energy gradient of the cost operator wrt the mixer pool at an ansatz layer specified by
            the input 'state' and initial point.
            Returns: The mixer operator with the largest energy gradient along with the associated energy
                     gradient."""

        if not isinstance(cost_operator, MatrixOp):
            cost_operator = MatrixOp(Operator(cost_operator.to_matrix()))

        if not ansatz:
            ansatz = self.initial_state
        # set parameters in ansatz


        param_dict = dict(zip(self._ansatz_params, self.best_gamma + [self.gamma_init] + self.best_beta + [self.beta_init] ))  # type: Dict
        wave_function = self.ansatz.assign_parameters(param_dict)

        #construct expectation operator
        exp_hc = (self.gamma_init*cost_operator).exp_i()
        exp_hc, exp_hc_ad = [_.to_matrix() for _ in [exp_hc, exp_hc.adjoint()]]
        energy_grad_op = exp_hc_ad@(commutator(1j*cost_operator,mixer).to_matrix())@exp_hc

        
        expectation = ExpectationFactory.build(
                operator= PrimitiveOp(energy_grad_op),
                backend=self.quantum_instance,
                include_custom=self._include_custom,
            )
        observable_meas = expectation.convert(StateFn(energy_grad_op, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()

        return expect_op, param_dict


    def _test_mixer_pool(self, operator:OperatorBase):
        #works off current self.optimal_mixer_pool
        energy_gradients = []

        for mixer in self.mixer_pool:
            new_mixer_list = self.optimal_mixer_list + [mixer]
            check_ansatz = self._check_operator_ansatz(operator, mixer_list = new_mixer_list)
            #parameterise ansatz
            expect_op, param_dict = self.compute_energy_gradient(mixer,operator, check_ansatz)
            #run expectation circuit
            sampled_expect_op = self._circuit_sampler.convert(expect_op, params= param_dict)
            meas = np.abs(np.real(sampled_expect_op.eval()))
            energy_gradients.append(meas)
        return self.mixer_pool[np.argmax(energy_gradients)]


    def _check_mixer_pool(self, mixer_pool:Optional[Union[str,List]] = None):
        if mixer_pool is None:
            mixer_pool = self.mixer_pool
        if isinstance(mixer_pool,str):
            mixer_pool = translate_mixer_str_opn(mixer_pool)
        elif isinstance(mixer_pool,Iterable):
            for mixer in mixer_pool:
                # TODO do some checks here
                pass
        else:
            # TODO raise correct error
            print(f"Type of {type(mixer_pool)} is not understood")
        
        self.mixer_pool = mixer_pool


    def constuct_adapt_ansatz(self, operator: OperatorBase) -> OperatorBase:
        energy_norm, p = 0, 0
        while energy_norm > self.threshold and p<self._reps:
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
        pass
    

    def run_adapt(self, operator:OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None):
        # main loop
        layer_reps = 0
        while layer_reps < self.max_reps:
            best_mixer = self._test_mixer_pool(operator)
            self.optimal_mixer_list.append(best_mixer)

            
            #perform optimisation of circuit:

        


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
