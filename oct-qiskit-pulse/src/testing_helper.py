import csv
from math import sqrt
from qiskit.quantum_info import random_hermitian
from qiskit.pulse.macros import measure_all
import datetime
import matplotlib.pyplot as plt
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qiskit.pulse import Play, Waveform
from qutip.qip.device import Processor
from qiskit import IBMQ
from src.helper import *
from src.qutip_helper import convert_qutip_ham
import qutip.control.pulsegen as pulsegen
import qutip.control.pulseoptim as cpo
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor, hadamard_transform, basis
from qutip.qip.algorithms import qft
import qutip.logging_utils as logging
logger = logging.get_logger()
# Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
# QuTiP control modules


example_name = 'QFT'


def get_hamiltonians(config, subsystem_list, wq0s):
    for wq in wq0s:
        config.hamiltonian['vars'][wq] = 1e-3

    ham = convert_qutip_ham(config, subsystem_list)
    H_d = ham['H_d']
    H_c = list(ham['H_c'].values())
    return(H_d, H_c)


def get_hadamard():
    h = hadamard_transform(1).full()
    hadamard_3 = Qobj([[h[0][0], h[0][1], 0],
                       [h[1][0], h[1][1], 0],
                       [0,      0,      1]])
    return hadamard_3


def get_sigmax():
    sigmax_3 = Qobj([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 1]])
    return sigmax_3

def get_gen_sigmax():
    a = Qobj([[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]])
    adag = Qobj([[0, 0, 0], [1, 0, 0], [0, sqrt(2), 0]])
    return a + adag


def get_sigmaz():
    return Qobj([[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, 1]])


def get_sigmay():
    sigmay_3 = Qobj([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
    return sigmay_3


def get_gensx():
    a = Qobj([[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]])
    adag = Qobj([[0, 0, 0], [1, 0, 0], [0, sqrt(2), 0]])
    return a + adag


def get_gensy():
    a = Qobj([[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]])
    adag = Qobj([[0, 0, 0], [1, 0, 0], [0, sqrt(2), 0]])

    return complex(0, 1) * (adag - a)


def get_cnot():
    return d_sum(Qobj(d_sum(identity(3), get_sigmax())), Qobj(identity(3)))

# def get_random():
#     random_hermitian(2)
#     new_output =


def plot_results(n_evo_times, n_ctrls, results, evo_times):
    fig1 = plt.figure(figsize=(30, 8))
    for i in range(n_evo_times):
        # Initial amps
        ax1 = fig1.add_subplot(2, n_evo_times, i+1)
        ax1.set_title("Init amps T={}".format(evo_times[i]))
        # ax1.set_xlabel("Time")
        ax1.get_xaxis().set_visible(False)
        if i == 0:
            ax1.set_ylabel("Control amplitude")
        for j in range(n_ctrls):
            ax1.step(results[i].time,
                     np.hstack((results[i].initial_amps[:, j],
                                results[i].initial_amps[-1, j])),
                     where='post')

        ax2 = fig1.add_subplot(2, n_evo_times, i+n_evo_times+1)
        ax2.set_title("Final amps T={}".format(evo_times[i]))
        ax2.set_xlabel("Time")
        # Optimised amps
        if i == 0:
            ax2.set_ylabel("Control amplitude")
        for j in range(n_ctrls):
            ax2.step(results[i].time,
                     np.hstack((results[i].final_amps[:, j],
                                results[i].final_amps[-1, j])),
                     where='post')

    plt.tight_layout()
    plt.show()


def run_optimizer(optim, dt, n_evo_times, dyn, p_type, n_ts, n_ctrls, f_ext, evo_time, evo_times, p_gen):
    results = []
    for i in range(n_evo_times):
        # Generate the tau (duration) and time (cumulative) arrays
        # so that it can be used to create the pulse generator
        # with matching timeslots
        dyn.init_timeslots()
        if i > 0:
            # Create a new pulse generator for the new dynamics
            p_gen = pulsegen.create_pulse_gen(p_type, dyn)

        # Generate different initial pulses for each of the controls
        init_amps = np.zeros([n_ts, n_ctrls])
        if (p_gen.periodic):
            phase_diff = np.pi / n_ctrls
            for j in range(n_ctrls):
                init_amps[:, j] = p_gen.gen_pulse(start_phase=phase_diff*j)
        # if alg == 'CRAB':
            # for j in range(n_ctrls)
        elif (isinstance(p_gen, pulsegen.PulseGenLinear)):
            for j in range(n_ctrls):
                p_gen.scaling = float(j) - float(n_ctrls - 1)/2
                init_amps[:, j] = p_gen.gen_pulse()
        elif (isinstance(p_gen, pulsegen.PulseGenZero)):
            for j in range(n_ctrls):
                p_gen.offset = sf = float(j) - float(n_ctrls - 1)/2
                init_amps[:, j] = p_gen.gen_pulse()
        else:
            # Should be random pulse
            for j in range(n_ctrls):
                init_amps[:, j] = p_gen.gen_pulse()

        dyn.initialize_controls(init_amps)

        # Save initial amplitudes to a text file
        if f_ext is not None:
            pulsefile = "ctrl_amps_initial_" + f_ext
            dyn.save_amps(pulsefile)
            print("Initial amplitudes output to file: " + pulsefile)

        print("***********************************")
        print("\n+++++++++++++++++++++++++++++++++++")
        print("Starting pulse optimisation for T={}".format(evo_time))
        print("+++++++++++++++++++++++++++++++++++\n")
        result = optim.run_optimization()
        # Save final amplitudes to a text file
        if f_ext is not None:
            pulsefile = "ctrl_amps_final_" + f_ext
            dyn.save_amps(pulsefile)
            print("Final amplitudes output to file: " + pulsefile)

        # Report the results
        result.stats.report()
        print("Final evolution\n{}\n".format(result.evo_full_final))
        print("********* Summary *****************")
        print("Final fidelity error {}".format(result.fid_err))
        print("Final gradient normal {}".format(result.grad_norm_final))
        print("Terminated due to {}".format(result.termination_reason))
        print("Number of iterations {}".format(result.num_iter))
        print("Completed in {} HH:MM:SS.US".format(
            datetime.timedelta(seconds=result.wall_time)))
        results.append(result)
        if i+1 < len(evo_times):
            # reconfigure the dynamics for the next evo time
            evo_time = evo_times[i+1]
            n_ts = int(float(evo_time) / dt)
            dyn.tau = None
            dyn.evo_time = evo_time
            dyn.num_tslots = n_ts
    return results


def run_qutip_sim(subsystem_list, backend, pulse_seq, H_c, H_d, def_seq=None, init_state=0):
    processor = Processor(N=1, dims=[3 ** len(subsystem_list)])

    dt = backend.configuration().dt * 1e9
    default = True
    # H_d = wq0 * (1 - sigmaz()) / 2
    tlist = np.array([dt * i for i in range(len(pulse_seq) + 1)])
    if not def_seq:
        for i, control in enumerate(H_c):
            processor.add_control(control, targets=0)
            coef = [a[i] for a in pulse_seq]

            processor.pulses[i].coeff = coef
            processor.pulses[i].tlist = tlist
    else:
        tlist = np.array([dt * i for i in range(len(def_seq) + 1)])
        processor.add_control(H_c[0], targets=0)
        processor.pulses[0].coeff = def_seq
        processor.pulses[0].tlist = tlist

    # processor.pulses[0].coeff = seq_x
    # processor.pulses[0].tlist = np.array([dt * i for i in range(len(seq_x)+1)])
    # processor.add_control(H_c[0], targets=[0], label='sigmax')
    # processor.add_control(H_d, targets=[0], label="drift")
    processor.add_drift(H_d, targets=[0])

    basis0 = basis(3 ** len(subsystem_list), init_state)
    # basis0 = basis(4,0) + basis(4,2)
    result = processor.run_state(init_state=basis0)

    result.states[-1].tidyup(1.e-3)
    return result

# def_cnot = backend.defaults().instruction_schedule_map.get('', [0,1])
# freq in ghz and i in nanosec so should be fine


def derotate(amp, i, backend):
    dt = backend.configuration().dt * 1e9
    t = dt * (i)
    freq = backend.configuration().hamiltonian['vars']['wq0']
    return np.exp(complex(0, -1) * freq * t)*amp


def generate_qiskit_program(pulse_seq, backend, default_seq=None, derotate_flag=False, draw_progs=False):

    D0 = []
    D1 = []
    if default_seq:
        D0 = default_seq
    else:
        for i, a in enumerate(pulse_seq):
            if derotate_flag:
                amp2 = derotate(a[1], i, backend)
                amp1 = derotate(a[0], i, backend)
                D0.append(complex(amp1, amp2))
            else:
                if len(pulse_seq[0]) == 2:
                    # D0.append(complex(a[0], a[1]))
                    D0.append(complex(a[0], a[1]))
                elif len(pulse_seq[0]) == 4:
                    D0.append(complex(a[0], a[1]))
                    D1.append(complex(a[2], a[3]))
                else:
                    D0.append(a[0])
                # if len(pulse_seq[0]) == 2:
                #     # D0.append(complex(a[0], a[1]))
                #     D0.append(complex(a[0], a[1]))
                #     D1.append(complex(a[2], a[3]))
                # else:
                #     D0.append(a[0])
            # D0.append(a[0])
            # D1.append(complex(a[2], a[3]))
    q1 = 0
    q2 = 1

    drive_chan1 = pulse.DriveChannel(q1)
    meas_chan1 = pulse.MeasureChannel(q1)
    acq_chan1 = pulse.AcquireChannel(q1)
    con_chan1 = pulse.ControlChannel(q1)

    drive_chan2 = pulse.DriveChannel(q2)
    meas_chan2 = pulse.MeasureChannel(q2)
    acq_chan2 = pulse.AcquireChannel(q2)
    con_chan2 = pulse.ControlChannel(q2)

    schedule = pulse.Schedule(name='sigmax')

    # init_x = backend.defaults().instruction_schedule_map.get('x', [1])
    # def_seq = init_x.instructions[0]
    # def_seq = def_seq[1].pulse.get_waveform().samples

    # init_x = backend.defaults().instruction_schedule_map.get('x').data
    # schedule += Play(Waveform(def_seq), drive_chan2)
    # schedule += init_x

    later = schedule.duration

    # schedule += def_cx << later
    # if len(subsystem_list) == 1:
    # schedule += Play(SamplePulse(pulse_seqs[0]), drive_chan1) << later
    # else:
    # schedule += Play(SamplePulse(pulse_seqs[0]), con_chan1) << later
    # schedule += Play(SamplePulse(pulse_seqs[1]), con_chan2) << later

    # schedule += Play(SamplePulse(pulse_seqs[2]), drive_chan1) << later

    # schedule += Play(SamplePulse(pulse_seqs[3]), drive_chan2)
    schedule += Play(Waveform(D0), drive_chan1) << later  # schedule.duration
    if len(pulse_seq[0]) == 4:
        schedule += Play(Waveform(D1), drive_chan2) << later  # schedule.duration
    schedule += measure_all(backend) << schedule.duration
    from qiskit import assemble

    if draw_progs:
        schedule.draw(plot_range=[0,len(D0)])

    num_shots = 1024
    qoc_test_program = assemble(schedule,
                                backend=backend,
                                meas_level=2,
                                meas_return='single',
                                #    qubit_lo_freq=backend.defaults().qubit_freq_est,
                                shots=num_shots,
                                )
    #    schedule_los=schedule_frequencies)
    return qoc_test_program

def generate_schedule(pulse_seq, backend, draw_progs):
    D0 = []
    D1 = []
    for i, a in enumerate(pulse_seq):
        if len(pulse_seq[0]) == 2:
            # D0.append(complex(a[0], a[1]))
            D0.append(complex(a[0], a[1]))
        elif len(pulse_seq[0]) == 4:
            D0.append(complex(a[0], a[1]))
            D1.append(complex(a[2], a[3]))
        else:
            D0.append(a[0])
            # if len(pulse_seq[0]) == 2:
            #     # D0.append(complex(a[0], a[1]))
            #     D0.append(complex(a[0], a[1]))
            #     D1.append(complex(a[2], a[3]))
            # else:
            #     D0.append(a[0])
        # D0.append(a[0])
        # D1.append(complex(a[2], a[3]))
    q1 = 0
    q2 = 1

    drive_chan1 = pulse.DriveChannel(q1)
    meas_chan1 = pulse.MeasureChannel(q1)
    acq_chan1 = pulse.AcquireChannel(q1)
    con_chan1 = pulse.ControlChannel(q1)

    drive_chan2 = pulse.DriveChannel(q2)
    meas_chan2 = pulse.MeasureChannel(q2)
    acq_chan2 = pulse.AcquireChannel(q2)
    con_chan2 = pulse.ControlChannel(q2)

    schedule = pulse.Schedule(name='sigmax')

    # init_x = backend.defaults().instruction_schedule_map.get('x', [1])
    # def_seq = init_x.instructions[0]
    # def_seq = def_seq[1].pulse.get_waveform().samples

    # init_x = backend.defaults().instruction_schedule_map.get('x').data
    # schedule += Play(Waveform(def_seq), drive_chan2)
    # schedule += init_x

    later = schedule.duration

    # schedule += def_cx << later
    # if len(subsystem_list) == 1:
    # schedule += Play(SamplePulse(pulse_seqs[0]), drive_chan1) << later
    # else:
    # schedule += Play(SamplePulse(pulse_seqs[0]), con_chan1) << later
    # schedule += Play(SamplePulse(pulse_seqs[1]), con_chan2) << later

    # schedule += Play(SamplePulse(pulse_seqs[2]), drive_chan1) << later

    # schedule += Play(SamplePulse(pulse_seqs[3]), drive_chan2)
    schedule += Play(Waveform(D0), drive_chan1) << later  # schedule.duration
    if len(pulse_seq[0]) == 4:
        schedule += Play(Waveform(D1), drive_chan2) << later  # schedule.duration
    schedule += measure_all(backend) << schedule.duration

    if draw_progs:
        schedule.draw(plot_range=[0,len(D0)])
    return schedule



def write_results(f_csv, acc, p_type, backend, gate_type):
    f = open(f_csv, 'wb')
    acc['p_type'] = p_type
    acc['backend'] = backend.name()
    acc['gate_type'] = gate_type
    w = csv.DictWriter(f, acc.keys())
    w.writerow(acc)
    f.close()
