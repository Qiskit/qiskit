
import csv
from src.testing_helper import generate_qiskit_program, get_hadamard, get_hamiltonians, get_sigmax, plot_results, run_optimizer
from qutip import hadamard_transform, sigmax, sigmay, identity
from qutip.tensor import tensor
import qutip.control.pulseoptim as cpo

import qutip.logging_utils as logging
import numpy as np
from src.helper import qubit_distribution


class QOCTester():
    # NOTE only supports ops on q0 or q0 and q1
    def __init__(self, config, subsystem_list, p_type='GAUSSIAN', two_level=False):
        self.zero_vars = ['wq0']
        self.omegad_mult = False
        self.two_level = two_level
        self.p_type = p_type
        self.subsystem_list = subsystem_list
        self.derotate = False
        self.config = config
        self.H_d, self.H_c = get_hamiltonians(self.config, self.subsystem_list, self.zero_vars)
        self.U_0 = identity(3 ** len(subsystem_list))
        self.n_ctrls = len(self.H_c)
        self.f_ext = None
        self.notes = ''
        if two_level:
            self.make_two_level()


    def make_two_level(self, last_two_qubits=True):
        self.two_level = True
        if not last_two_qubits:
            raise NotImplementedError
        omegad0 = self.config.hamiltonian['vars']['omegad0']
        if len(self.subsystem_list) == 2:
            omegad1 = self.config.hamiltonian['vars']['omegad1']

            q1sigx = tensor(sigmax(), identity(2))
            q1sigy = tensor(sigmay(), identity(2))
            q0sigx = tensor(identity(2), sigmax())
            q0sigy = tensor(identity(2), sigmay())
            self.H_c = [omegad0 * q0sigx, omegad0 *
                        q0sigy, omegad1 * q1sigx, omegad1 * q1sigy]
            self.U_0 = tensor(identity(2), identity(2))
        else:
            self.H_c = [omegad0 * sigmax(), omegad0 * sigmay()]
            self.U_0 = identity(2)
        self.H_d = self.U_0 * 0

    def init_unitaries(self, target_name, U_targ=None):
        if not U_targ:
            if target_name == 'x' or target_name == 'sigmax' or target_name == 'XGate':
                if self.two_level:
                    U_targ = sigmax()
                else:
                    U_targ = get_sigmax()
            elif target_name == 'h' or target_name == 'hadamard':
                if self.two_level:
                    U_targ = hadamard_transform(1)
                else:
                    U_targ = get_hadamard()
            elif target_name == 'cnot':
                if self.two_level:
                    U_targ = cnot()
                else:
                    U_targ = get_cnot()
        self.U_targ = U_targ
        self.target_name = target_name

    def time_init(self, n_ts_list=None, dt=None):
        if not dt:
            dt = self.config.dt * 1e9
        self.dt = dt
        if n_ts_list is None:
            n_ts_0 = list(np.arange(64, 160, 32))
            n_ts_1 = list(np.arange(160, 1600, 160))
            n_ts_list = n_ts_0 + n_ts_1
        evo_times = [dt * n for n in n_ts_list]
        n_evo_times = len(evo_times)
        evo_time = evo_times[0]
        n_ts = int(float(evo_time) / dt)
        self.evo_times = evo_times
        self.evo_time = evo_time
        self.n_evo_times = n_evo_times
        self.n_ts = n_ts
        self.n_ts_list = n_ts_list

    def create_pulse_optimizer(self, fid_err_targ=1e-30, max_iter=2000, max_wall_time=120, min_grad=1e-30, log_level=logging.INFO):
        optim = cpo.create_pulse_optimizer(self.H_d, self.H_c, self.U_0, self.U_targ, self.n_ts, self.evo_time,
                                           amp_lbound=-.7, amp_ubound=.7,
                                           fid_err_targ=fid_err_targ, min_grad=min_grad,
                                           max_iter=max_iter, max_wall_time=max_wall_time,
                                           optim_method='fmin_l_bfgs_b',
                                           method_params={'max_metric_corr': 20,
                                                          'accuracy_factor': 1e8},
                                           dyn_type='UNIT',
                                           fid_params={'phase_option': 'PSU'},
                                           init_pulse_type=self.p_type,
                                           log_level=log_level, gen_stats=True, alg='GRAPE')
        return optim

    def tester_run_optimizer(self, optim):
        optim.test_out_files = 0
        dyn = optim.dynamics
        dyn.test_out_files = 0
        p_gen = optim.pulse_generator
        f_ext = self.f_ext
        results = run_optimizer(optim, self.dt, self.n_evo_times, dyn, self.p_type, self.n_ts,
                                self.n_ctrls, f_ext, self.evo_time, self.evo_times, p_gen)
        return results

    def plot_all_results(self, results):
        plot_results(self.n_evo_times, self.n_ctrls, results, self.evo_times)

    def gen_qiskit_progs(self, results, backend, draw_progs=False):
        return [generate_qiskit_program(result.final_amps, backend, derotate_flag=self.derotate, draw_progs=draw_progs) for result in results]

    def run_qiskit_programs(self, programs, backend):
        jobs = []
        for i, program in enumerate(programs):
            cur_job = backend.run(program)
            jobs.append(cur_job)
            pulse_steps = self.n_ts_list[i]
            cur_job.update_tags(['QOC', self.target_name, 'n_ts=' + str(pulse_steps), self.p_type])
        self.jobs = jobs
        return jobs

    def get_acc(self, jobs):
        last_qub = self.config.n_qubits

        acc = {a: 0 for a in self.n_ts_list}
        for i, job in enumerate(jobs):
            n_ts = self.n_ts_list[i]
            cur_acc = qubit_distribution(job.result().get_counts())[last_qub-1][1]/1024
            acc[n_ts] = cur_acc
        return acc

    def write_acc(self, acc, filename, header=False):
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['n_ts', 'frac_1', 'p_type', 'backend',
                          'gate_type', 'two_lvl', 'delta0', 'creation_date', 'notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if header:
                writer.writeheader()
            for i, row in enumerate(acc.keys()):
                out_dict = {'n_ts': row, 'frac_1': acc[row], 'p_type': self.p_type, 'backend': self.config.backend_name,
                            'gate_type': self.target_name, 'two_lvl': self.two_level, 'delta0': self.config.hamiltonian['vars']['delta0'], 'creation_date': self.jobs[i].creation_date(), 'notes': self.notes}
                writer.writerow(out_dict)

# TODO Don't forget to put in stuff for derotate, omegad_mult, etc
