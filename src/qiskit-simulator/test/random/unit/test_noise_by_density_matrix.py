#!/usr/bin/python3

'''
This test creates a circuit which consists of three qubits, initialized to a random state,
and one id gate on one of the qubits.
The test attaches random noise to the id gate, and calculates the expected density matrix at the end of execution.
It then invokes the simulator, and compares the result with the expected density matrix.
This process repeats for several types of noise.

In the future, we will perform the same checks for additional gates types.
'''


import importlib
import argparse
import random
import math
import numpy as np
import json    
import qiskit.backends._qiskit_cpp_simulator as qsim
from qiskit.tools.qi.pauli import pauli_group
import qiskit.tools.qi.qstructs as qs
serv = importlib.import_module('src.qiskit-simulator.test.service_functions_for_tests', None)


nqubits = 3
shots = 100000
tol = 0.05


class ChannelTest:
    """
    A base class for different types of noise.
    This class creates a noisy circuit (see its description at the top of this file),
    simulates it, and verifies correctness of the result.
    """
    
    def __init__(self, interfaces, 
                 initial_state, qubit, channel_name):
        """
        Initializes parameters related to circuit creation.

        Right now only interface 'native' is supported by the simulator.
        Explanations about the 'interface' argument will follow
        when the simulator is enhanced with operator-sum capabilities.

        'initial_state' is ideally of type qs.QuantumState.
        Note that qs.QuantumState has an option to create a random state.

        'qubit' is the index of the qubit where the noisy id gate is located.
        """
        self.circuits = []
        self.initial_state = initial_state
        self.interfaces = interfaces
        self.channel_name = channel_name
        self.operators = []
        self.qubit = qubit


    def calc_expected_density_matrix(self):
        """
        Calculates the density matrix after the circuit execution.
        
        Basically this means applying the noise,
        which is stored in 'operators' (operator-sum representation),
        on the given qubit
        (this of course affects all the qubits).
        """
        return qs.DensityMatrix(states = self.initial_state).single_qubit_noise(qubit = self.qubit, operators = self.operators)


    def create_basic_circuit(self):
        """
        Returns the circuit segment that does not contain noise
        """
        basic_circuit = {
            'header': {
                'number_of_qubits': nqubits,
                'number_of_clbits': 0,
                'qubit_labels': [['q'+str(i), i] for i in range(nqubits)],
                'clbit_labels': [],
                },
            'operations': [
                {'name': 'id', 'qubits': [self.qubit]}
                ]
            }
        return basic_circuit  


    def create_noise_params(self, type_of_interface):
        """
        Returns the circuit segment of the noise,
        i.e., 'noise_params' in 'config'.

        Currently only type_of_interface = 'native'
        is supported by the simulator.
        """
        noise_params = self.noise_params_all_gates(type_of_interface)
        gate_noise = self.noise_params_id_gate(type_of_interface)
        if len(gate_noise) > 0:
           noise_params['id'] = gate_noise

        return noise_params


    def noise_params_all_gates(self, type_of_interface):
        """
        Returns the part of 'noise_params'
        which is global to all types of gates
        """
        if type_of_interface == 'operator_sum':
            return {}
        elif type_of_interface == 'native':
            return self.noise_params_native_all_gates()


    def noise_params_id_gate(self, type_of_interface):
        """
        Returns the part of 'noise_params'
        which is specific to the id gate
        """
        if type_of_interface == 'operator_sum':
            return {'operator_sum': np.reshape(np.array(
                                    [[np.real(y), np.imag(y)] 
                                    for op in self.operators 
                                    for x in op 
                                    for y in x]), 
                                    (len(self.operators), 2,2,2)) 
                                            }
        elif type_of_interface == 'native':
            return self.noise_params_native_id_gate()


    def create_qobj(self):
        """
        Returns the entire circuit, including noise
        """
        qobj = {'id': self.channel_name,
                'config': {
                    'shots': shots,
                    'max_threads_shot': 1 + np.random.randint(2),
                    'max_threads_gate': 1 + np.random.randint(2),
                    'data': ['density_matrix'],
                    'initial_state':[[np.real(amplitude), np.imag(amplitude)] for amplitude in self.initial_state.amplitudes]
                    }
                }

        qobj['circuits'] = []
        for interface in self.interfaces:
            qobj['circuits'].append({'name': interface,
                                     'compiled_circuit': self.create_basic_circuit()})

            noise_params = self.create_noise_params(interface)
            if(len(noise_params) > 0):
                qobj['circuits'][-1]['config'] = {'noise_params': noise_params}                    
            
        return qobj


    def execute(self, seed, executable):
        """
        Executes the simulator and verifies correctness
        """
        qobj = self.create_qobj()

        if seed is not None:
            qobj['config']['seed'] = seed
        
        result = qsim.run(qobj, executable = executable)

        assert result['success'] == True, \
               'Simulator execution failed. Execution details: ' + str(result)
        
        self.verify(result)
        return result
    

    def verify(self, sim_result):
        """
        Verifies that the density matrix in the simulator's output
        is close to the expected density matrix
        """
        expected = self.calc_expected_density_matrix()
        assert all(qs.is_close(expected.rho, res['data']['density_matrix'], rel_tol = tol, abs_tol = tol) for res in sim_result['result']), \
               'Simulator verification failed. Execution details: ' + str(sim_result) + ', expected density matrix: ' + str(expected.rho)

        

class IdealChannelTest(ChannelTest):
    """
    No noise channel
    """

    def __init__(self, interfaces, initial_state, qubit):
        ChannelTest.__init__(self, interfaces = interfaces, initial_state = initial_state,
                             qubit = qubit, channel_name = 'ideal')
        self.operators = [ np.identity(2) ]

    def noise_params_native_all_gates(self):
        return {}

    def noise_params_native_id_gate(self):
        return {}



class PauliChannelTest(ChannelTest):
    
    def __init__(self, interfaces, initial_state, qubit, probabilities):
        ChannelTest.__init__(self, initial_state = initial_state, interfaces = interfaces,
                             qubit = qubit, channel_name = 'pauli')
        assert len(probabilities) == 4, \
               'PauliChannel: invalid probabilities vector: ' + str(probabilities)
        self.probabilities = probabilities.probs
        self.operators = [math.sqrt(p)*m for (p,m) in zip(probabilities.probs, [pauli.to_matrix() for pauli in pauli_group(1,1)])]

    def noise_params_native_all_gates(self):
        return {}

    def noise_params_native_id_gate(self):
        return {'p_pauli': self.probabilities}



class RelaxationTest(ChannelTest):
    
    def __init__(self, interfaces, initial_state, qubit,
                 rate, thermal):
        ChannelTest.__init__(self, interfaces = interfaces, initial_state = initial_state,
                             qubit = qubit, channel_name = 'relax')
        self.rate = rate
        self.thermal = thermal.probs
        assert qs.is_close(1, math.fsum(self.thermal))

        error_prob = 1 - math.e**(-self.rate)
        self.operators = [ math.sqrt(1-error_prob)*np.identity(2) ]
        self.operators.extend([ math.sqrt(error_prob*self.thermal[i])* 
                                np.outer(qs.QuantumState.basic_state(i).amplitudes, qs.QuantumState.basic_state(j).amplitudes.conj()) 
                                for i in range(2) for j in range(2) ])      


    def noise_params_native_all_gates(self):
        return {'relaxation_rate': self.rate, 'thermal_populations': self.thermal}

    def noise_params_native_id_gate(self):
        return { 'gate_time': 1 } 



class AmplitudeDampingTest(ChannelTest):
    
    def __init__(self, interfaces, initial_state, qubit,
                 gamma):
        ChannelTest.__init__(self, interfaces = interfaces, initial_state = initial_state,
                             qubit = qubit, channel_name = 'amplitude_damping')

        self.operators = [
            [[1, 0], [0, math.sqrt(1-gamma)]],
            [[0, math.sqrt(gamma)], [0, 0]]
            ]


class Category2Test(ChannelTest):
    """
    We will provide documentation to this channel
    when we add operator-sum noise to the simulator
    """
    
    def __init__(self, interfaces, initial_state, qubit): 
        ChannelTest.__init__(self, interfaces = interfaces, initial_state = initial_state,
                             qubit = qubit, channel_name = 'category_2')

        self.operators = [
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, -0.5], [-0.5, 0.5]]
            ]



class UerrorTest(ChannelTest):
    def __init__(self, interfaces, initial_state, qubit,
                 u_err):
        ChannelTest.__init__(self, interfaces = interfaces, initial_state = initial_state,
                             qubit = qubit, channel_name = 'u_err')
        self.u_err = u_err
        self.operators = [u_err.mat]


    def noise_params_native_all_gates(self):
        return {}

    def noise_params_native_id_gate(self):
        return { 'U_error':
                 np.reshape(np.array([[np.real(y), np.imag(y)] for x in self.u_err.mat for y in x]), (2,2,2))
                 }


class DepolarizingTest(ChannelTest):
    def __init__(self, interfaces, initial_state, qubit,
                 p_depol):
        ChannelTest.__init__(self, interfaces = interfaces, initial_state = initial_state,
                             qubit = qubit, channel_name = 'depol')
        self.p_depol = p_depol
        self.operators = PauliChannelTest(probabilities = qs.ProbabilityDistribution([1-3*p_depol/4, p_depol/4, p_depol/4, p_depol/4]),
                                          interfaces = interfaces,
                                          initial_state = initial_state,
                                          qubit = qubit).operators


    def noise_params_native_all_gates(self):
        return {}

    def noise_params_native_id_gate(self):
        return { 'p_depol': self.p_depol } 

   

if __name__=='__main__':

    parser = serv.parse(description = 'Simulator noise unit test')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(a = args.seed)
        np.random.seed(args.seed)

    for i in range(50):
        selected = np.random.randint(5)
        if selected==0:
            channel = IdealChannelTest(interfaces = ['native'],
                                       initial_state = qs.QuantumState(nqubits = nqubits),
                                       qubit = np.random.randint(nqubits))
        elif selected==1:
            channel = PauliChannelTest(interfaces = ['native'], initial_state = qs.QuantumState(nqubits = nqubits),
                                       qubit = np.random.randint(nqubits),
                                       probabilities = qs.ProbabilityDistribution(n=4))
        elif selected==2:
            channel = RelaxationTest(interfaces = ['native'], initial_state = qs.QuantumState(nqubits = nqubits),
                                     qubit = np.random.randint(nqubits),
                                     rate = np.random.rand(1)[0], 
                                     thermal = qs.ProbabilityDistribution(n=2))
        elif selected==3:
            channel = UerrorTest(interfaces = ['native'], initial_state = qs.QuantumState(nqubits = nqubits),
                                 qubit = np.random.randint(nqubits),
                                 u_err = qs.UnitaryOperation())
        else:
            channel = DepolarizingTest(interfaces = ['native'], initial_state = qs.QuantumState(nqubits = nqubits),
                                       qubit = np.random.randint(nqubits),
                                       p_depol = np.random.rand(1)[0])

        channel.execute(args.seed, args.executable)
