from functools import reduce
from itertools import combinations_with_replacement, permutations, product
from qiskit.transpiler.passes.optimization.template_matching import maximal_matches
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from numpy.random.mtrand import hypergeometric
from qiskit import QuantumCircuit, algorithms
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.opflow import ExpectationBase, I, OperatorBase, X, Y, Z, H, PauliSumOp
from qiskit.opflow.expectations.expectation_factory import ExpectationFactory
from qiskit.opflow.gradients import GradientBase
from qiskit.opflow.primitive_ops import MatrixOp
from qiskit.opflow.primitive_ops import primitive_op
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.providers import Backend, BaseBackend
from qiskit.quantum_info import Operator
from qiskit.utils import QuantumInstance, algorithm_globals

from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOA
from qiskit.algorithms.minimum_eigen_solvers.vqe import _validate_initial_point
from qiskit.circuit.quantumregister import QuantumRegister
# from .qaoa import QAOA

from random import randint


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
        mixer_pool: Optional[Union[OperatorBase, QuantumCircuit]] = None,
        mixer_pool_type: Optional[str] = None,
        threshold: Optional[
            Callable[[int, float], None]
        ] = None,  # todo: add default value for threshold
        max_reps = 1,
        **kwargs,
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
            threshold: A positive, real value in which the algorithm stops once the norm of the gradient is below this threshold.
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
        self._exp_seed = randint(1,100)
        super().__init__(**kwargs)
        self.threshold = threshold

        if mixer_pool is not None and mixer_pool_type is not None:
            raise AttributeError(
                "A custom mixer pool can be passed in or a mixer pool type can be passed in but not both"
            )

        self.mixer_pool = mixer_pool
        self.mixer_pool_type = mixer_pool_type
        self.max_reps = max_reps

        self.optimal_mixer_list = []  # will be appending optimal mixers to this, first mixer is H see above
        self.reps = 1               # Overload QAOA's reps parameter & set it to 1.
        self.name = 'AdaptQAOA'

        if self.mixer_pool_type is None:
            self.mixer_pool = mixer_pool  # todo: check if this list of operators/circuits needs more preprocessing before use
        if self.threshold is None:
            self.threshold = 0.001      # TODO: work out a way to better set a default threshold

        self.initial_point[0] = 0.01    # Set an initial value for gamma according to https://arxiv.org/abs/2005.10258
        self.best_gamma = [self.initial_point[0]]

    def _check_operator_ansatz(self, operator: OperatorBase) -> OperatorBase:
        # Recreates a circuit based on operator parameter.
        if operator != self._cost_operator:
            self._cost_operator = operator
            self.ansatz = QAOAAnsatz(
                operator, initial_state=self._initial_state, mixer_operator=self.optimal_mixer_list,
                name=self.name
            ).decompose()  # TODO remove decompose once #6674 is fixed

    def compute_energy_gradient(self, mixer, cost_operator, ansatz=None):
        from qiskit.opflow import commutator

        """Computes the energy gradient of the cost operator wrt the mixer pool at an ansatz layer specified by
            the input 'state' and initial point.
            Returns: The mixer operator with the largest energy gradient along with the associated energy
                     gradient."""

        if not isinstance(cost_operator, MatrixOp):
            cost_operator = MatrixOp(Operator(cost_operator.to_matrix()))
        self.cost_operator = cost_operator
        
        wave_function = ansatz.assign_parameters(self.hyperparameter_dict)
        # construct expectation operator
        exp_hc = (self.best_gamma[-1] * cost_operator).exp_i() 
        exp_hc_ad = exp_hc.adjoint().to_matrix()
        exp_hc = exp_hc.to_matrix()
        energy_grad_op = exp_hc_ad @ (commutator(cost_operator, mixer).to_matrix()) @ exp_hc
        energy_grad_op = PrimitiveOp(energy_grad_op)

        expectation = ExpectationFactory.build(
            operator=energy_grad_op,
            backend=self.quantum_instance,
            include_custom=self._include_custom,
        )
        observable_meas = expectation.convert(StateFn(energy_grad_op, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()

        return expect_op

    def _test_mixer_pool(self, operator: OperatorBase, ansatz: QuantumCircuit = None):
        self._check_problem_configuration(operator=operator)
        energy_gradients = []
        for mixer in self.mixer_pool:
            expect_op = self.compute_energy_gradient(mixer, operator, ansatz=self.ansatz)
            # run expectation circuit
            sampled_expect_op = self._circuit_sampler.convert(expect_op, params=self.hyperparameter_dict)
            meas = np.abs(np.real(sampled_expect_op.eval()))
            energy_gradients.append(meas)
        max_energy_idx = np.argmax(energy_gradients)
        print('test_mixer_pool ansatz')
        print(self.mixer_pool[max_energy_idx])
        return self.mixer_pool[max_energy_idx], energy_gradients[max_energy_idx]

    def constuct_adapt_ansatz(self, operator: OperatorBase) -> OperatorBase:
        self.cost_operator = operator
        p, self.ansatz = 0, self.initial_state  # initialise layer loop counter and ansatz
        while p < self.max_reps:        # loop over number of maximum reps
            best_mixer, energy_norm = self._test_mixer_pool(operator=operator)
            print("Current energy norm | Threshold  =====> | {} | {} |".format(energy_norm,self.threshold))
            if energy_norm < self.threshold:          # Threshold stoppage condition
                break
            self.optimal_mixer_list.append(best_mixer)  # Append mixer associated with largest energy gradient to list
            self.ansatz = QAOAAnsatz(
                                    cost_operator=operator, 
                                    initial_state=self._initial_state, 
                                    mixer_operator=self.optimal_mixer_list,
                                    name=self.name
                                )
            # print(self.ansatz.decompose().draw())
            result = self.compute_minimum_eigenvalue(operator=operator)#self.find_minimum(cost_fn = operator, ansatz = ansatz)
            # print(self.ansatz.decompose().draw())
            opt_params = result.optimal_point
            self.best_gamma = list(opt_params[0:][::2])
            self.best_beta = list(opt_params[1:][::2])
            print("Optimal parameters: {}".format(opt_params))
            print("Initial point: {}".format(self.initial_point))
            self._reps+=1
            p += 1
        pass

    def _check_problem_configuration(self, operator: OperatorBase):
        # Generates the pool of mixers with respect to the cost operator size
        self.num_qubits = operator.num_qubits # Compute the number of qubits from the problem operator
        # if not self.mixer_pool: # Set a generate a mixer pool if none was provided
        #     self.mixer_pool = adapt_mixer_pool(num_qubits=self.num_qubits, pool_type=self.mixer_pool_type)
        if isinstance(self.mixer_pool,list):
            mixer_n_qubits = [mixer.num_qubits for mixer in self.mixer_pool]
        else:
            mixer_n_qubits = self.mixer_pool.num_qubits
            self.mixer_pool = [self.mixer_pool]
        check_mixer_qubits = list(np.argwhere(mixer_n_qubits != self.num_qubits)[0])
        if check_mixer_qubits:
            err_str = ", ".join(map(lambda x: str(x), check_mixer_qubits))
            raise ValueError(
                "One or more mixing oeprators specified at list indices {}"
                " have an unequal number of respective qubits {} to the"
                " initialised cost operator {}.".format(
                    err_str, mixer_n_qubits[check_mixer_qubits], self.num_qubits
                )
            )
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
    def mixer_pool(self) -> List:
        if not self._mixer_pool:
            try:
                self._mixer_pool = adapt_mixer_pool(
                    num_qubits=self.num_qubits, pool_type=self.mixer_pool_type
                )
            except:  # TODO: Fix this num_qubits exception
                self._mixer_pool = adapt_mixer_pool(num_qubits=5, pool_type=self.mixer_pool_type)
        return self._mixer_pool

    @mixer_pool.setter
    def mixer_pool(self, mixer_pool: List) -> None:
        self._mixer_pool = mixer_pool

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Returns an optional initial state as a circuit"""
        if self._initial_state is not None:
            return self._initial_state

        # if no initial state is passed and we know the number of qubits, then initialize it.
        if self.num_qubits > 0:
            initial_state = QuantumCircuit(self.num_qubits)
            initial_state.h(range(self.num_qubits))
            return initial_state
        # otherwise we cannot provide a default
        return None

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """Sets initial state."""
        self._initial_state = initial_state
        self._invalidate()

    @property
    def hyperparameter_dict(self) -> Dict:
        self._hyperparameter_dict = {}
        if self._ansatz_params:
            reordered_params = list(np.zeros(len(self._ansatz_params)))
            reordered_params[0:][::2] = self.best_gamma
            reordered_params[1:][::2] = self.best_beta
            self._hyperparameter_dict = dict(
                zip(
                    self._ansatz_params,
                    self.best_gamma + self.best_beta,
                )
            )
        return self._hyperparameter_dict

    @hyperparameter_dict.setter
    def hyperparameter_dict(self, hyperparameter_dict) -> Dict:
        self._hyperparameter_dict = hyperparameter_dict

    @property
    def cost_operator(self):
        """Returns an operator representing the cost of the optimization problem.

        Returns:
            OperatorBase: cost operator.
        """
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        """Sets cost operator & number of qubits for optimization problem.

        Args:
            cost_operator (OperatorBase, optional): cost operator to set.
        """
        self.num_qubits = cost_operator.num_qubits
        self._cost_operator = cost_operator
    
    @property
    def initial_point(self):
        if self._ansatz_params:
            self.ansatz.parameter_bounds = [(-2 * np.pi, 2 * np.pi)] * self.ansatz.num_parameters
            if len(self._initial_point)!=self.ansatz.num_parameters:
                self._update_initial_point()
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point) -> Optional[np.ndarray]:
        if initial_point is None:
            initial_point = self._generate_initial_point()
        self._initial_point = initial_point

    def _generate_initial_point(self, reps=1):
        return algorithm_globals.random.uniform(2 * reps * [-2 * np.pi], 2 * reps * [2 * np.pi])

    def _update_initial_point(self):
        self._initial_point = np.concatenate((self._initial_point,self._generate_initial_point()))

def adapt_mixer_pool(
    num_qubits: int, add_single: bool = True, add_multi: bool = True, pool_type: str = None
) -> List:
    """
    Gets all combinations of mixers in desired set (standard qaoa mixer, single qubit mixers, multi qubit mixers)
    Args:
        num_qubits: number of qubits
        add_single: whether to add single qubit to mixer pool (not standard qaoa x mixers)
        add_multi: whether to add multi qubit to mixer pool
        pool_type: Optional input overrides add_single and add_multi by respecifying these conditions based on
                   the preset mixer pool classes: 'multi', 'singular' and 'single'.

    Returns:
        List of all possible combinations of mixers.
    """
    if pool_type:
        if pool_type == "multi":
            add_multi, add_single = True, True
        elif pool_type == "singular":
            add_multi, add_single = False, True
        elif pool_type == "single":
            add_multi, add_single = False, False
        else:
            raise ValueError(
                "Unrecognised mixer pool type {}, modify this input to the available presets"
                " 'single', 'singular' or 'multi'."
            )

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


if __name__ == "__main__":
    from functools import reduce
    from qiskit.opflow import I, X, Y, Z
    from qiskit.algorithms import QAOA
    from qiskit import Aer
    from qiskit.utils import QuantumInstance

    QISKIT_DICT = {"I": I, "X": X, "Y": Y, "Z": Z}

    def string_to_qiskit(qstring):
        if is_all_same(qstring):
            # case where its all X's or Y's
            gate = qstring[0]
            list_string = [
                i * "I" + gate + (len(qstring) - i - 1) * "I" for i in range(len(qstring))
            ]
            return sum(
                [
                    reduce(lambda a, b: a ^ b, [QISKIT_DICT[char.upper()] for char in x])
                    for x in list_string
                ]
            )

        return reduce(lambda a, b: a ^ b, [QISKIT_DICT[char.upper()] for char in qstring])

    def is_all_same(items):
        return all(x == items[0] for x in items)

    mixer_list = ["XXIII", "XIIX", "IXXII"]
    cost_op = string_to_qiskit("IIIZZ") + string_to_qiskit("ZZIII")
    mixer_list = [string_to_qiskit(x) for x in mixer_list]

    quantum_instance = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=1024)

    adapt = AdaptQAOA(mixer_pool_type="singular", max_reps=5, quantum_instance=quantum_instance)
    print(adapt.constuct_adapt_ansatz(cost_op))

    # qaoa = QAOA(reps=5, quantum_instance=quantum_instance)
    # out = qaoa.compute_minimum_eigenvalue(cost_op)
    # print(out)
