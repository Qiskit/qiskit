from qiskit.providers.aer.pulse.hamiltonian_model import HamiltonianModel
from qutip import identity, sigmax, sigmaz, sigmay, hadamard_transform
import qutip.control.pulseoptim as cpo
import qutip.logging_utils as logging
import numpy as np

from qutip.qip.device import Processor
from qutip.states import basis
from qutip.operators import identity


def pulse_ham_qutip(backend):
    config = backend.configuration()
    subsystem_list = list(range(config.n_qubits))
    ham_string = config.hamiltonian
    hamiltonian = HamiltonianModel.from_dict(ham_string, subsystem_list)

    H_d = identity(2)
    H_c = [identity(2)]

    # hamiltonian = backend.configuration().
    return H_d, H_c


def qutip_optimize_wrapper(wq0, omegad0, evo_time, n_ts, target='sigmax', final_evo=False, phase=None, p_type='RND'):
    # Drift Hamiltonian
    # H_d = wq0 * sigmaz() #wq0 * (1-sigmaz())/2 / omegad0

    H_d = wq0 * (1 - sigmaz()) / 2
    # H_d = wq0 * np.sin(np.pi/3) [sigmax]
    # The (single) control Hamiltonian

    H_c = [omegad0 * sigmax(), omegad0 * sigmay()]

    # start point for the gate evolution
    U_0 = identity(2)
    if phase:
        H_d = 0 * sigmax()
        # H_c = [np.cos(phase) * sigmax + np.sin(phase) * sigmay()]
        # d(t)*H_c = H(t)
        # d(t) [cos(phase(t)) * sigmax() + np.sin(phase) * sigmay()] = H(t)
        # f(t) = d(t) * cos(phase(t))
        #
        # [d(t) cos(phase(t)) * omegad0 * sigmax(), d(t) sin(phase(t)) * omegad0 * sigmay()]
        # [f(t) * omegad0 * sigmax(), f1(t) * omegad0 * sigmay()]
        # f(t) = d(t) * cos(phase(t))
        # f1(t) = d(t) * sin(phase(t))
        #
        # D(t) = f(t) + i * f1(t)
        # '''drive is Re[d(t)]
        # absolute value of D(t) is d(t)
        #select complex amplitudes and take real and complex amplitudes
    # Target for the gate evolution X gate
    if target == 'sigmax':
        U_targ = sigmax()
    elif target == 'sigmay':
        U_targ = sigmay()
    elif target == 'hadamard':
        U_targ = hadamard_transform(1)
    else:
        raise NotImplementedError

    logger = logging.get_logger()
    # Set this to None or logging.WARN for 'quiet' execution
    log_level = logging.WARN

    # Fidelity error target
    fid_err_targ = 1e-10
    # Maximum iterations for the optimisation algorithm
    max_iter = 2000
    # Maximum (elapsed) time allowed in seconds
    max_wall_time = 120
    # Minimum gradient (sum of gradients squared)
    # as this tends to 0 -> local minima has been found
    min_grad = 1e-20  # Fidelity error target

    # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
    p_type = p_type

    result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time,
                                        fid_err_targ=fid_err_targ, min_grad=min_grad,
                                        max_iter=max_iter, max_wall_time=max_wall_time,
                                        out_file_ext=None, init_pulse_type=p_type,
                                        log_level=log_level, gen_stats=True, amp_lbound=-1, amp_ubound=1)

    if final_evo:
        print("Final evolution\n{}\n".format(result.evo_full_final))

    return [amp[0] for amp in result.final_amps], result


def qutip_simulate_wrapper(pulse_seq, dt, omegad0, wq0, print_result=False):
    processor = Processor(N=1)

    H_d = wq0 * (1 - sigmaz()) / 2

    processor.add_control(H_d, targets=0, label="sigmaz")
    processor.add_control(omegad0 * sigmay(), targets=0, label="sigmay")
    processor.add_control(omegad0 * sigmax(), targets=0, label="sigmax")

    coef = pulse_seq
    tlist = np.array([dt * i for i in range(len(coef) + 1)])

    processor.pulses[0].coeff = coef
    processor.pulses[0].tlist = tlist
    # for pulse in processor.pulses:
    # pulse.print_info()

    basis0 = basis(2, 0)
    result = processor.run_state(init_state=basis0)
    # result.states[-1].tidyup(1.e-5)
    # result.states
    # basis(2,0)

    if print_result:
        print(result.states[-1].tidyup(1.e-5))

    return result
