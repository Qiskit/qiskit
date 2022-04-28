from functools import reduce
from itertools import combinations_with_replacement, permutations, product
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
#from qiskit.algorithms import QAOA
from .qaoa import QAOA
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import ExpectationBase, OperatorBase
from qiskit.opflow.expectations.expectation_factory import ExpectationFactory
from qiskit.opflow.gradients import GradientBase
from qiskit.opflow.primitive_ops import MatrixOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.providers import Backend, BaseBackend
from qiskit.quantum_info import Operator
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.opflow import I, X, Y, Z

# CHANGEBACK TODO
from ...circuit.library.n_local.qaoa_ansatz import QAOAAnsatz


class AdaptQAOA(QAOA):
    """
    The Adaptive Derivative Assembled Problem Tailored - Quantum Approximate Optimization Algorithm.

    `ADAPT-QAOA <https://arxiv.org/abs/2005.10258>` __ is a variation of the well-known algorithm for finding solutions to combinatorial-optimization problems.

    The ADAPT-QAOA implementation directly extends :class:`QAOA` and inherits QAOA's optimization structure.
    However, unlike QAOA, which has a fixed form of the ansatz, ADAPT-QAOA takes an iterative approach
    to finding a more optimal ansatz for the given problem.

    An optional array of :math:`2p` parameter values, as the *initial_point*, may be provided as the
    starting **beta** and **gamma** parameters (as identically named in the
    original `QAOA paper <https://arxiv.org/abs/1411.4028>`__) for the ADAPT-QAOA ansatz.

    A list of operators or parameterized quantum circuits may optionally also be provided as a custom
    `mixer_pool`. The build options for the mixer pool contains the standard single-qubit X rotations
    and single-qubit Y mixers as well as the option of also including multi-qubit entangling gates.
    """

    def __init__(
        self,
        optimizer: Optimizer = None,
        max_reps: int = 5,
        initial_state: Optional[QuantumCircuit] = None,
        gamma_init: Optional[float] = 0.01,
        beta_init: Optional[float] = np.pi / 4,
        mixer_pool: Optional[Union[OperatorBase, QuantumCircuit]] = None,
        mixer_pool_type: Optional[str] = "multi",
        threshold: Optional[
            Callable[[int, float], None]
        ] = None,  # todo: add default value for threshold
        gradient: Optional[Union[GradientBase, Callable[[Union[np.ndarray, List]], List]]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        max_evals_grouped: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """
        Args:
            optimizer: A classical optimizer.
            max_reps: An optional maximum number of repetitions of the ADAPT-QAOA circuit (defaults to 5).
            initial_state: An optional initial state to prepend the ADAPT-QAOA circuit with.
            gamma_init: An optional initial value for the parameter gamma to use as a starting value for the optimizer.
            beta_init: An optional initial value for the parameter beta to use as a starting value for the optimizer.
            mixer_pool: An optional custom list of Operators or QuantumCircuits that make up a pool from which mixers are chosen from.
                Cannot be used in conjunction with `mixer_pool_type`.
            mixer_pool_type: An optional string representing different mixer pool types `single` creates the same mixer pool as the
                standard QAOA. `singular` creates a mixer pool including mixers in `single` as well as additional single qubit
                mixers. `multi` creates a mixer pool including mixers from `single`, `singular` as well as multi-qubit entangling mixers.
                Cannot be used in conjuction with `mixer_pool`.
            threshold: A value in which the algorithm stops once the norm of the gradient is below this threshold.
            gradient: An optional gradient operator respectively a gradient function used for
                      optimization.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When None (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to True (defaults to False).
            include_custom: When `expectation` parameter here is None setting this to True will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Ignored if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
            quantum_instance: Quantum Instance or Backend
        """

        super().__init__(
            optimizer=optimizer,
            reps=max_reps,
            initial_state=initial_state,
            gradient=gradient,
            expectation=expectation,
            include_custom=include_custom,
            max_evals_grouped=max_evals_grouped,
            callback=callback,
            quantum_instance=quantum_instance,
        )
        self.gamma_init, self.beta_init = gamma_init, beta_init
        self.threshold = threshold
        self.mixer_pool = mixer_pool
        self.mixer_pool_type = mixer_pool_type
        self.optimal_mixer_list = []  # --------------> Will be appending optimal mixers to this.
        self.cost_operator = None
        self.max_reps = max_reps
        self.best_gamma = []
        self.best_beta = []
        self.num_qubits = 5  # TODO: change this to num qubits, should read directly from cost operator

        if self.mixer_pool is not None and self.mixer_pool_type is not None:
            raise AttributeError(
                "A custom mixer can be passed in or a mixer pool type can be passed in but not both"
            )
        if self.mixer_pool_type == "multi":
            self.mixer_pool = adapt_mixer_pool(
                num_qubits=self.num_qubits, add_single=True, add_multi=True
            )
        elif self.mixer_pool_type == "singular":
            self.mixer_pool = adapt_mixer_pool(
                num_qubits=self.num_qubits, add_single=True, add_multi=False
            )
        elif self.mixer_pool_type == "single":
            self.mixer_pool = adapt_mixer_pool(
                num_qubits=self.num_qubits, add_single=False, add_multi=False
            )
        elif self.mixer_pool_type is None:
            self.mixer_pool = mixer_pool  # todo: check if this list of operators/circuits needs more preprocessing before use

    def _check_operator_ansatz(self, operator: OperatorBase, mixer_list=None) -> OperatorBase:
        # Recreates a circuit based on operator parameter.
        if mixer_list is None:
            mixer_list = self.optimal_mixer_list

        self.ansatz = QAOAAnsatz(
            operator, self._reps, initial_state=self._initial_state, mixer_operator=mixer_list
        )

        self._ansatz_params = sorted(self.ansatz.parameters, key=lambda p: p.name)

        return self.ansatz

    def compute_energy_gradient(self, mixer, cost_operator, ansatz):
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

        param_dict = dict(
            zip(
                self._ansatz_params,
                self.best_gamma + [self.gamma_init] + self.best_beta + [self.beta_init],
            )
        )  # type: Dict
        wave_function = self.ansatz.assign_parameters(param_dict)
        # construct expectation operator
        exp_hc = (self.gamma_init * cost_operator).exp_i()
        exp_hc, exp_hc_ad = [_.to_matrix() for _ in [exp_hc, exp_hc.adjoint()]]
        energy_grad_op = exp_hc_ad @ (commutator(1j * cost_operator, mixer).to_matrix()) @ exp_hc
        energy_grad_op = PrimitiveOp(energy_grad_op)

        expectation = ExpectationFactory.build(
            operator=energy_grad_op,
            backend=self.quantum_instance,
            include_custom=self._include_custom,
        )
        observable_meas = expectation.convert(StateFn(energy_grad_op, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()

        return expect_op, param_dict

    def _test_mixer_pool(self, operator: OperatorBase):
        # works off current self.optimal_mixer_pool
        energy_gradients = []

        for mixer in self.mixer_pool:
            new_mixer_list = self.optimal_mixer_list + [mixer]
            check_ansatz = self._check_operator_ansatz(operator, mixer_list=new_mixer_list)
            # parameterise ansatz
            expect_op, param_dict = self.compute_energy_gradient(mixer, operator, check_ansatz)
            # run expectation circuit
            sampled_expect_op = self._circuit_sampler.convert(expect_op, params=param_dict)
            meas = np.abs(np.real(sampled_expect_op.eval()))
            energy_gradients.append(meas)
        return self.mixer_pool[np.argmax(energy_gradients)], np.max(energy_gradients)

    def constuct_adapt_ansatz(self, operator: OperatorBase) -> OperatorBase:
        energy_norm, p = 0, 0
        while energy_norm > self.threshold and p < self._reps:
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
        """ ""
        pass

    def run_adapt(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ):
        # main loop
        self.num_qubits = operator.num_qubits
        layer_reps = 0
        terminate = False
        while layer_reps < self.max_reps and terminate == False:
            best_mixer, energy = self._test_mixer_pool(operator)
            if energy < self.threshold:
                terminate = True
            self.optimal_mixer_list.append(best_mixer)
            # perform optimisation of circuit:
            self.compute_minimum_eigenvalue(operator)
            

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
        mixer_pool += [i * "I" + "X" + (num_qubits - i - 1) * "I" for i in range(num_qubits)]
        mixer_pool += [i * "I" + "Y" + (num_qubits - i - 1) * "I" for i in range(num_qubits)]
    if add_multi:
        indicies = list(permutations(range(num_qubits), 2))
        indicies = list(set(tuple(sorted(x)) for x in indicies))
        combos = list(combinations_with_replacement(["X", "Y", "Z"], 2))
        full_multi = list(product(indicies, combos))
        for item in full_multi:
            iden_str = list("I" * num_qubits)
            iden_str[item[0][0]] = item[1][0]
            iden_str[item[0][1]] = item[1][1]
            mixer_pool.append("".join(iden_str))

    op_dict = {"I": I, "X": X, "Y": Y, "Z": Z}

    def is_all_same(items):
        return all(x == items[0] for x in items)

    def string_to_op(qstring):
        if is_all_same(qstring):
            # case where its all X's or Y's
            gate = qstring[0]
            list_string = [
                i * "I" + gate + (len(qstring) - i - 1) * "I" for i in range(len(qstring))
            ]
            return sum(
                [
                    reduce(lambda a, b: a ^ b, [op_dict[char.upper()] for char in x])
                    for x in list_string
                ]
            )

        return reduce(lambda a, b: a ^ b, [op_dict[char.upper()] for char in qstring])

    mixer_pool = [string_to_op(mixer) for mixer in mixer_pool]
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
