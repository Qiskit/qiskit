#!/usr/bin/python3

"""
This is a system test for the qiskit simulator.
Its purpose is:
1) Verify that there are no fatal errors and segmentation faults.
2) Test performance.

This test generates a random circuit.
It then applies noise on the gates (partly random, partly hard-coded), and runs the simulator on the obtained circuit.
Finally, it verifies that throughout the simulator's execution, the state vector's norm is always equal to 1.

Currently the test does not check correctness.
"""


import importlib
import argparse
import random
import numpy as np
import qiskit.tools.qi.qstructs as qs
serv = importlib.import_module('src.qiskit-simulator.test.service_functions_for_tests', None)



def check_result(result):

# TODO: add a check that a qubit is either |0> or |1> after measurement

    if result['success'] == True:
        for shot in result['result'][0]['data']['saved_quantum_states']:
            for j in range(ngates):
                state_vec = shot[j]
                calculated_norm = np.linalg.norm(state_vec)
                assert qs.is_close(1, calculated_norm), \
                       'Result state vector has norm ' + str(calculated_norm) + \
                       'State vector is: ' + str(state_vec)
        

if __name__=='__main__':
    
    parser = serv.parse(description = 'Simulator noise system test')
    args = parser.parse_args()

    nqubits = random.randint(8, 15)
    ngates = random.randint(100, 200)  # total number of gates, spread across the qubits

    qobj = {'id': 'test_self_check',
            'config': {
                'shots': 10,
                'data': ['saved_quantum_states'],
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
                        },
                    'CX': {
                        'U_error': qs.UnitaryOperation(n=2).mat
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

        ops.append({"name": "save", "qubits": [0], "params": [j]})

    result = serv.run_simulator(qobj, args.executable)
    check_result(result)
