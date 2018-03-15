#!/usr/bin/python3

'''
This test demonstrates run time improvement, following an optimization in the simulator.
The optimization refrains from applying an operator on the state vector, if the operator is the identity matrix.
'''

import importlib
import argparse
import random
import numpy as np
serv = importlib.import_module('src.qiskit-simulator.test.service_functions_for_tests', None)


if __name__=='__main__':

    parser = serv.parse(description = 'Simulator noise system test')
    args = parser.parse_args()

    nqubits = random.randint(10, 20)
    ngates = random.randint(200, 500)  # total number of gates, spread across the qubits

    qobj = {'id': 'test_optimization',
            'config': {
                'shots': 10,
                'max_threads_shot': 2,
                'max_threads_gate': 2,
                'noise_params': {
                    'id': {}
                    }
                },
            'circuits': [
                {
                    'name': 'some_circuit',
                    'compiled_circuit': {
                        'header': {
                            'number_of_qubits': nqubits, 
                            'number_of_clbits': 0, 
                            'qubit_labels': [['q', i] for i in range(nqubits)],
                            'clbit_labels': []
                            }, 
                        'operations': [
                            {'name': 'id', 'qubits': [np.random.randint(nqubits)]}
                            for j in range(ngates)
                            ]
                        }
                    }
                ]
            }

    if args.seed is not None:
        qobj['config']['seed'] = args.seed   

    # U_error as identity
    qobj['config']['noise_params']['id'] = {'U_error': [[1, 0], [0, 1]]}
    serv.run_simulator(qobj, args.executable)
    
    # Pauli selects identity
    qobj['config']['noise_params']['id'] = {'p_pauli': [0.9, 0.1, 0, 0]}
    serv.run_simulator(qobj, args.executable)
    
    # No coherent or Pauli error, some relaxation error       
    qobj['config']['noise_params']['id'] = {'gate_time': 1}
    qobj['config']['noise_params']['relaxation_rate'] = 0.2
    serv.run_simulator(qobj, args.executable)
