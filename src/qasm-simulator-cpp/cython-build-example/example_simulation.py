import numpy as np
import json
from qasm_simulator import SimulatorWrapper


class SimulatorEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        return json.JSONEncoder.default(self, obj)


class SimulatorDecoder(json.JSONDecoder):
    """
    JSON decoder for the output from C++ qasm_simulator.

    This converts complex vectors and matrices into numpy arrays
    for the following keys.
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        for key in ['U_error', 'density_matrix']:
            # JSON is a complex matrix
            if key in obj:
                tmp = np.array(obj[key])
                obj[key] = tmp[::, ::, 0] + 1j * tmp[::, ::, 1]
        for key in ['quantum_state', 'inner_products']:
            # JSON is a list of complex vectors
            if key in obj:
                for j in range(len(obj[key])):
                    tmp = np.array(obj[key][j])
                    obj[key][j] = tmp[::, 0] + 1j * tmp[::, 1]
        return obj


def cpp_run(qobj_dict):
    """
    Execute a qobj dictionary on the C++ QASM simulator.

    Args:
        qobj (dict): a python dictionary qobj.

    Returns:
        result (dict): the simulator output.

    Note that this will serialize complex Numpy arrays into valid JSON
    and deserialize corresponding json back into complex Numpy arrays.
    """
    simulator = SimulatorWrapper()
    result = simulator.run(json.dumps(qobj_dict, cls=SimulatorEncoder))
    return json.loads(result, cls=SimulatorDecoder)


# Example qobj
qobj = {
    "id": "cython_test",
    "config": {
        "shots": 1000,
        "seed": 1,
        "backend": "local_qasm_simulator_cpp"
    },
    "circuits": [
        {
            "name": "bell measure",
            "compiled_circuit": {
                "header": {
                    "clbit_labels": [["c", 2]],
                    "number_of_clbits": 2,
                    "number_of_qubits": 2,
                    "qubit_labels": [["q", 0], ["q", 1]]
                },
                "operations": [
                    {"name": "h", "qubits": [0]},
                    {"name": "cx", "qubits": [0, 1]},
                    {"name": "snapshot", "params": [0]},
                    {"name": "measure", "qubits": [0], "clbits": [0]},
                    {"name": "measure", "qubits": [1], "clbits": [1]}
                ]
            }
        },
        {
            "name": "bell measure",
            "config": {
                "noise_params": {
                    "CX": {"p_pauli": [0.1 / 16 for _ in range(15)]},
                    "X90": {"p_pauli": [0.01 / 4 for _ in range(3)]}
                }
            },
            "compiled_circuit": {
                "header": {
                    "clbit_labels": [["c", 2]],
                    "number_of_clbits": 2,
                    "number_of_qubits": 2,
                    "qubit_labels": [["q", 0], ["q", 1]]
                },
                "operations": [
                    {"name": "h", "qubits": [0]},
                    {"name": "cx", "qubits": [0, 1]},
                    {"name": "measure", "qubits": [0], "clbits": [0]},
                    {"name": "measure", "qubits": [1], "clbits": [1]}
                ]
            }
        }
    ]
}


# Execute and print result
result = cpp_run(qobj)
print(result)
