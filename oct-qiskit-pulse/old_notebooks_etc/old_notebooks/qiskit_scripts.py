import copy
import time

import numpy as np
# run_QTP()

import qiskit
from qiskit.ignis.verification import process_tomography_circuits, ProcessTomographyFitter
from qiskit.providers.aer.pulse import PulseSystemModel
from qiskit import IBMQ, pulse, assemble, Aer, QuantumRegister
from qiskit.pulse import Play
from qiskit.quantum_info import state_fidelity, process_fidelity
from qiskit.tools.monitor import job_monitor

from qiskit.circuit import Gate


def run_opt_pulse(pulse_seq, backend, sim=True, num_shots=1024, freq_est=0):
    schedule = pulse.Schedule(name="excited state")

    opt_pi_pulse = pulse.SamplePulse(pulse_seq, name="optimized x")
    acquire = pulse.Acquire(100, channel=pulse.AcquireChannel(
        0), mem_slot=pulse.MemorySlot(0), name='acquire')

    inst_sched_map = backend.defaults().instruction_schedule_map
    measure = inst_sched_map.get(
        'measure', qubits=backend.configuration().meas_map[0])

    drive_chan = pulse.DriveChannel(0)

    schedule += Play(opt_pi_pulse, drive_chan)
    # schedule.draw()

    if sim:
        schedule += acquire << schedule.duration
        program = assemble(schedule, backend=backend, meas_level=2,
                           meas_return='single', shots=num_shots, qubit_lo_freq=[freq_est])
        aer_model = PulseSystemModel.from_backend(backend)

        aer_backend = qiskit.providers.aer.PulseSimulator(
            configuration=backend.configuration())

        job = aer_backend.run(program, aer_model)

    else:
        schedule += measure << schedule.duration
        program = assemble(schedule, backend=backend,
                           meas_level=2, meas_return='single', shots=num_shots, qubit_lo_freq=[freq_est])

        job = backend.run(program)

    job_monitor(job)
    # print(sum(job.result(timeout=120).get_memory(0)))
    return job, schedule


def pulse_to_gate_1q(backend, input_pulse: pulse.SamplePulse):
    schedule = pulse.Schedule(name='new 1q gate')
    schedule += input_pulse(pulse.DriveChannel(0))

    circ_inst_map = backend.defaults().circuit_instruction_map
    map_copy = copy.deepcopy(circ_inst_map)
    map_copy.add('opt_gate', 0, schedule)

    opt_gate = Gate('opt_gate', 1, [])
    return opt_gate, map_copy


def run_qpt_gate_1q(backend, input_gate: Gate, default_gate=None, my_inst_map=None, num_shots=4000):
    q = QuantumRegister(1)
    circ = qiskit.QuantumCircuit(q)
    if default_gate == 'x':
        circ.x(0)
    elif default_gate == 'h':
        circ.h(0)
    else:
        circ.append(input_gate, [0])

    # circ = transpiled_circ
    # job = qiskit.execute(circ, Aer.get_backend('unitary_simulator'))
    # ideal_unitary = job.result().get_unitary(circ)

    # Generate process tomography circuits and run on qasm simulator
    qpt_circs = process_tomography_circuits(circ, q)

    if default_gate:
        job = qiskit.execute(qpt_circs, backend, shots=num_shots)
        # ideal_unitary = job.result().get_unitary(circ)
    else:

        transpiled_circ = qiskit.transpile(qpt_circs, basis_gates=backend.configuration().basis_gates + ['opt_gate'])

        qpt_sched = qiskit.schedule(transpiled_circ, inst_map=my_inst_map, backend=backend)

        job = qiskit.execute(qpt_sched, backend, inst_map=my_inst_map, shots=num_shots)
        ideal_unitary = None

        # Extract tomography data so that counts are indexed by measurement configuration
    # qpt_tomo = ProcessTomographyFitter(job.result(), qpt_circs)

    # real_job = qiskit.execute(new_sched, armonk_backend, shots=4000, inst_map=my_inst_map, basis_gates=backend.configuration().basis_gates+['opt_gate'])

    return job, circ

def run_fidelity(qpt_tomo, choi_ideal):
    # MLE Least-Squares tomographic reconstruction
    t = time.time()
    choi_lstsq = qpt_tomo.fit(method='lstsq')
    print('Least-Sq Fitter')
    print('fit time:', time.time() - t)
    print('fit fidelity (state):', state_fidelity(choi_ideal / 2, choi_lstsq.data / 2))
    print('fit fidelity (process):', np.real(process_fidelity(choi_ideal, choi_lstsq.data, require_cptp=False)))

    # CVXOPT Semidefinite-Program tomographic reconstruction
    t = time.time()
    choi_cvx = qpt_tomo.fit(method='cvx')
    print('\nCVXOPT Fitter')
    print('fit time:', time.time() - t)
    print('fit fidelity (state):', state_fidelity(choi_ideal / 2, choi_cvx.data / 2))
    print('fit fidelity (process):', np.real(process_fidelity(choi_ideal, choi_cvx.data, require_cptp=False)))

