from qiskit import QiskitError
try:
    from qiskit import Aer
    from qiskit.providers.aer import noise
except ImportError:
    raise QiskitError("Unable to generate mitigation data without Aer simulator")

# For simulation
import qiskit.utils.mitigation as mit
from qiskit import QuantumRegister
from qiskit.result import Result
from qiskit.test import QiskitTestCase

def generate_mitigation_matrices(num_qubits, sim, noise_model, method='tensored'):
    seed_simulator = 100
    shots = 10000

    qr = QuantumRegister(num_qubits)
    qubit_list = range(num_qubits)
    meas_calibs, state_labels = mit.complete_meas_cal(
        qubit_list=qubit_list, qr=qr, circlabel="mcal"
    )
    cal_res = sim.run(
        meas_calibs,
        shots=shots,
        seed_simulator=seed_simulator,
        basis_gates=noise_model.basis_gates,
        noise_model=noise_model,
    ).result()

    if method == 'complete':
        meas_fitter = mit.CompleteMeasFitter(
            cal_res, state_labels, qubit_list=qubit_list, circlabel="mcal"
        )
        return meas_fitter.cal_matrix

    elif method == 'tensored':
        mit_pattern = [[qubit] for qubit in qubit_list]
        meas_fitter = mit.TensoredMeasFitter(
            cal_res, mit_pattern=mit_pattern, circlabel="mcal"
        )
        return meas_fitter.cal_matrices

    return None


def generate_data(num_qubits, circuits, method = 'tensored', noise_model = None):
    sim = Aer.get_backend("aer_simulator")
    matrices = generate_mitigation_matrices(num_qubits, sim, noise_model, method=method)
    print(matrices)

def readout_errors_1(num_qubits):
    # Create readout errors
    readout_errors = []
    for i in range(num_qubits):
        p_error1 = (i + 1) * 0.002
        p_error0 = 2 * p_error1
        ro_error = noise.ReadoutError(
            [[1 - p_error0, p_error0], [p_error1, 1 - p_error1]])
        readout_errors.append(ro_error)

    # Readout Error only
    noise_model = noise.NoiseModel()
    for i in range(num_qubits):
        noise_model.add_readout_error(readout_errors[i], [i])
    return noise_model

num_qubits = 4
noise_model = readout_errors_1(num_qubits)
circuits = []
method = 'tensored'
generate_data(num_qubits, circuits, method = method, noise_model = noise_model)


