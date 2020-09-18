# This is the qutip version of base optimizer

import qutip.logging_utils as logging
#
from qiskit.circuit import gate
from qiskit.pulse import schedule, Schedule
from qutip.control.pulseoptim import optimize_pulse_unitary
#
from .QOCOptimizer import QOCOptimizer
from .qutip_helper import *
from .schedule_helper import qutip_amps_to_channels
from qutip import identity
from src.helper import d_sum


def results_amps_to_dict(amp_list, hamiltonian_keys):
    # NOTE this assumes same ordering
    out_dict = defaultdict(list)
    for element in amp_list:
        for i, val in enumerate(hamiltonian_keys):
            out_dict[val].append(element[i])
    return out_dict


class QutipOptimizer(QOCOptimizer):
    def __init__(self, backend, n_ts=None, p_type='RND', two_level=False, alg='GRAPE', zero_wq=True, log_level='WARN'):
        """initialize a QutipOptimizer

        Args: 
            backend ([type]): The backend of the system to be used. 
            n_ts ([type], optional): The number of time steps. Defaults to None.
            p_type (str, optional): The type of pulse initialization to use.
                Defaults to 'RND'. 
            two_level (bool, optional): Whether or not to use a two level model.
                Defaults to False. 
            alg (str, optional): The type of QOC algorithm to use. Defaults to 'GRAPE'. 
            zero_wq (bool, optional): Whether to perform the RWA and remove the rotating components of the drift hamiltonian. Defaults to True.
        """
        self.n_ts = n_ts
        self.dt = backend.configuration().dt
        self.config = backend.configuration()
        self.p_type = p_type
        self.two_level = two_level
        self.alg = alg
        self.zero_wq = zero_wq
        self.log_level = log_level

    def run_optimizer(self, unitary, qubit_targets, *args, fid_err_targ=1e-30, max_iter=2000, max_wall_time=120,
                      min_grad=1e-30, alg='GRAPE', zero_wq=True, **kwargs):

        if self.zero_wq:
            for i in qubit_targets:
                self.config.hamiltonian['vars']['wq' + str(i)] = 0
        hamiltonian = convert_qutip_ham(self.config, qubit_targets, two_level=self.two_level)
        drift_hamiltonian = hamiltonian['H_d']
        control_hamiltonians = hamiltonian['H_c']

        if self.n_ts:
            n_ts = self.n_ts
        else:
            n_ts = 640
        if self.two_level:
            U_0 = Qobj(identity(2 ** len(qubit_targets)).full())
        else:
            U_0 = Qobj(identity(3 ** len(qubit_targets)).full())

        U_targ = unitary
        p_type = self.p_type
        dt = self.dt
        evo_time = dt * n_ts * 1e9
        if self.log_level == 'WARN':
            log_level = logging.WARN
        elif self.log_level == 'INFO':
            log_level = logging.INFO
        else:
            log_level = logging.WARN

        result = optimize_pulse_unitary(drift_hamiltonian, list(control_hamiltonians.values()), U_0, U_targ, n_ts,
                                        evo_time,
                                        fid_err_targ=fid_err_targ, min_grad=min_grad,
                                        max_iter=max_iter, max_wall_time=max_wall_time,
                                        out_file_ext=None, init_pulse_type=p_type,
                                        log_level=log_level, gen_stats=True, alg=alg,
                                        amp_lbound=-0.7, amp_ubound=0.7, phase_option='PSU',
                                        optim_method='fmin_l_bfgs_b',
                                        method_params={'max_metric_corr': 20,
                                                       'accuracy_factor': 1e8},
                                        # dyn_type='UNIT',
                                        fid_params={'phase_option': 'PSU'},
                                        *args, **kwargs)
        amp_dictionary = results_amps_to_dict(result.final_amps, hamiltonian['H_c'].keys())
        return amp_dictionary, hamiltonian['H_c'].keys()

    def get_pulse_schedule(self, input_gate: gate, qubit_targets: list) -> schedule:
        """Get an optimized pulse schedule from a target unitary operator.

        Args:
            input_gate (gate): The input gate, which contains the desired operator.
            qubit_targets (list): The target qubits to use for the evolution

        Returns:
            schedule: The optimized pulse schedule
        """
        assert (len(qubit_targets) < 3)
        unitary = Qobj(input_gate.to_matrix())
        if not self.two_level:
            unitary = raise_unitary(unitary)
        result_amps, control_hamiltonians = self.run_optimizer(unitary, qubit_targets)
        return qutip_amps_to_channels(result_amps)

        # Maybe this ^^ should be class based, that way each optimizer is a class
