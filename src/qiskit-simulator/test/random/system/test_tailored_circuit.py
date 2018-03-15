#!/usr/bin/python3

"""
This is a system test for the qiskit simulator.
Its purpose is:
1) Verify that there are no fatal errors and segmentation faults.
2) Test performance.

This test generates a random circuit.
It then applies hard-coded noise on the gates, and runs the simulator on the obtained circuit.
Finally, it verifies that throughout the simulator's execution, the state vector's norm is always equal to 1.

This test can be used as a basis for more complex tests, like test_self_check.

Currently the test does not check correctness.
"""

import importlib
import argparse
import random
serv = importlib.import_module('src.qiskit-simulator.test.service_functions_for_tests', None)


if __name__=='__main__':

    parser = serv.parse(description = 'Simulator noise system test')
    args = parser.parse_args()

    nqubits = random.randint(15, 25)
    ngates = random.randint(200, 500)  # total number of gates, spread across the qubits

    qobj = {'id': 'system_test_1',
            'config': {
                'shots': 10,
                'max_threads_shot': 2,
                'max_threads_gate': 2,
                'noise_params': {
                    'relaxation rate': 0.1,
                    'thermal_populations': [0.2, 0.8],
                    'id': {
                        'p_depol': 0.1,
                        'p_pauli': [1, 2, 3, 4],
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
            'circuits': [
                {
                    'name': 'some_circuit',
                    'compiled_circuit': {
                        'header': {
                            'number_of_qubits': nqubits, 
                            'number_of_clbits': 1, 
                            'qubit_labels': [['q', i] for i in range(nqubits)],
                            'clbit_labels': [['c', 1]]
                            }, 
                        'operations': [
                            ]
                        }
                    }
                ]
            }

    if args.seed is not None:
        qobj['config']['seed'] = args.seed    
        
    for j in range(ngates):

        ops = qobj['circuits'][0]['compiled_circuit']['operations']
        gate_names = ['id', 'h', 's', 'cx', 'measure']
        g = random.randint(0, len(gate_names)-1)

        if g<=2:
            ops.append({'name': gate_names[g], 'qubits': [random.randint(0, nqubits-1)]})
        
        if gate_names[g] == 'cx':
            ops.append({'name': 'cx', 'qubits': list(random.sample(range(nqubits), 2))})
                
        if gate_names[g] == 'measure':
            qubit = random.randint(0, nqubits-1)
            ops.append({'name': 'measure', 'qubits': [qubit], 'clbits': [0]})

    serv.run_simulator(qobj, args.executable)
