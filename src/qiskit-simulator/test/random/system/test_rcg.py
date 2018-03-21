#!/usr/bin/python3

'''
This is a system test for the qiskit simulator.
Its purpose is:
1) Verify that there are no fatal errors and segmentation faults.
2) Test performance.

This test is using test.python._random_circuit_generator to generate a random circuit (hence the name test_rcg).
It then applies hard-coded noise on the gates, and runs the simulator on the obtained circuit.

Currently the test does not check correctness.
'''


import importlib
import argparse
import numpy as np
import test.python._random_circuit_generator as gen
serv = importlib.import_module('src.qiskit-simulator.test.service_functions_for_tests', None)

def create_qobj(seed = None):
    
    a = gen.RandomCircuitGenerator(min_qubits = 15, max_qubits = 25, min_depth = 200, max_depth = 300, seed = seed)

    # 'basis' is a list of gates which are supported both by the random circuit generator and the simulator
    a.add_circuits(n_circuits = 1,
                   basis = ['reset', 'x', 'y', 'z', 'h', 's', 't', 'u1', 'u2', 'u3', 'cx', 'cz'])
    
    b = a.get_circuits('qobj')

    qobj = {'id': 'test_rcg',
            'config': {
                'shots': 10,
                'max_threads_shot': 2,
                'max_threads_gate': 2,
                'noise_params': {
                    'relaxation_rate': 0.1,
                    'thermal_populations': [0.2, 0.8],
                    'id': {
                        'p_depol': 0.1,
                        'p_pauli': [1, 2, 3],
                        'U_error': [
                            [0, 1],
                            [1, 0]
                        ],
                        'gate_time': 0.3
                        },
                    'U': {
                        'p_depol': 0.2,
                        'p_pauli': [4, 3, 2, 1],
                        'U_error': [
                            [1, 0],
                            [0, -1]
                            ],
                        'gate_time': 2
                        }
                    }
                },
            'circuits': [{
                'name': 'circuit_1',
                'compiled_circuit': b[0]
                }]
            }

    if seed is not None:
        qobj['config']['seed'] = seed

    return qobj
    

if __name__=='__main__':

    parser = serv.parse(description = 'Simulator noise system test')
    args = parser.parse_args()
    
    qobj = create_qobj(args.seed)
    serv.run_simulator(qobj, args.executable)
