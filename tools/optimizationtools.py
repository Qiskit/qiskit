"""
Quantum Optimization tools.

These are simple tools that are used in our optimization examples

Author: Jay Gambetta
"""
import sys
import os
# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, ry, barrier, cz
from qiskit.simulators._simulatortools import enlarge_single_opt, enlarge_two_opt
import numpy as np




def cost_function(data, n, alpha, beta):
    """Compute the cost function.

    data  is a dictionary of the form data = {'00000': 10}
    n = number of qubits
    alpha is a vector with elements q0 -- qn

    beta is a matrix of couplings
    """

    temp = 0
    tot = sum(data.values())
    for key in data:
        observable = 0
        for j in range(len(key) - n, len(key)):
            if key[j] == '0':
                observable = observable + alpha[len(key) - 1 - j]
            elif key[j] == '1':
                observable = observable - alpha[len(key) - 1 - j]
            for i in range(j):
                if key[j] == '0' and key[i] == '0':
                    observable = observable + \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '1' and key[i] == '1':
                    observable = observable + \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '0' and key[i] == '1':
                    observable = observable - \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '1' and key[i] == '0':
                    observable = observable - \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
            for i in range(j + 1, len(key)):
                if key[j] == '0' and key[i] == '0':
                    observable = observable + \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '1' and key[i] == '1':
                    observable = observable + \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '0' and key[i] == '1':
                    observable = observable - \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '1' and key[i] == '0':
                    observable = observable - \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
        temp += data[key] * observable / tot
    return temp


def make_Hamiltonian(n, alpha, beta):
    """Compute the cost function.

    n = number of qubits
    alpha is a vector with elements q0 -- qn
    beta is a matrix of couplings
    WARNING. This is exponential in the number of qubits.
    """

    Hamiltonian = 0
    Z = np.array([[1, 0], [0, -1]])
    ZZ = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    for j in range(n):
        Hamiltonian += alpha[j]*enlarge_single_opt(Z, j, n)
        for i in range(0, j):
            Hamiltonian += beta[i, j]*enlarge_two_opt(ZZ, i, j, n)

    return Hamiltonian



def trial_funtion_optimization(n, m, theta, entangler_map):
    """Trial function for classical optimization problems.

    n = number of qubits
    m = depth
    theta = control vector of size n*m stacked as theta[n*i+j] where j counts
           the qubits and i the depth
    entangler_map = {0: [2, 1],
                     1: [2],
                     3: [2],
                     4: [2]}
    control is the key and values are the target
    """
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    trial_circuit.h(q)
    for i in range(m):
        trial_circuit.barrier(q)
        for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cz(q[node], q[j])
        for j in range(n):
            trial_circuit.ry(theta[n * i + j], q[j])
    trial_circuit.barrier(q)
    for j in range(n):
        trial_circuit.measure(q[j], c[j])
    return trial_circuit


def trial_funtion_optimization_no_meas(n, m, theta, entangler_map):
    """Trial function for classical optimization problems.

    n = number of qubits
    m = depth
    theta = control vector of size n*m stacked as theta[n*i+j] where j counts
           the qubits and i the depth
    entangler_map = {0: [2, 1],
                     1: [2],
                     3: [2],
                     4: [2]}
    control is the key and values are the target
    """
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    trial_circuit.h(q)
    for i in range(m):
        trial_circuit.barrier(q)
        for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cz(q[node], q[j])
        for j in range(n):
            trial_circuit.ry(theta[n * i + j], q[j])
    return trial_circuit

def trial_circuit_pauli(n,m,theta,entangler_map,pauli_string):

    # create a trial circuit with the final post-rotations that depend on pauli_string 
    
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    
    
    trial_circuit.h(q)
    for i in range(m):
           trial_circuit.barrier(q)
        
    for j in range(n):
            trial_circuit.ry(theta[n * i * 2 + 2*j], q[j])
            trial_circuit.rz(theta[n * i * 2 + 2*j + 1], q[j])        
        
        
    for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cz(q[node], q[j])
        

    for j in range(n):
        if pauli_string[j]=='X':
            trial_circuit.ry(-np.pi/2, q[j])
            
        elif pauli_string[j]=='Y':
            trial_circuit.rx(np.pi/2, q[j])

    for j in range(n):
        trial_circuit.measure(q[j], c[j])

    return trial_circuit




def parse_hamiltonian_file(file_name):
    
    #parses a Hamiltonian file into an array

    file = open(file_name,'r+')

    textAll = file.readlines()
    textAll = [x.strip() for x in textAll]
    

   
    dim = len(textAll)
    textStr=[]

    for i in range(dim):
        if (i % 2) == 0:
            textStr.append(textAll[i])
    
        else:
            Numb = float(textAll[i])
            textStr.append(Numb)
    
    return textStr

def text_to_ham(file_name):
    
    # Builds a Hamiltonian matrix out of a Hamiltonian text file 
    
    ham_array=parse_hamiltonian_file(file_name)
    hamiltonian=np.zeros(np.power(2,len(ham_array[0])),np.power(2,len(ham_array[0])))
    identity=np.matrix([[1,0],[0,1]])
    sx=np.matrix('0 1;1 0')
    sy=np.matrix('0 -1j;1j 0')
    sz=np.matrix('1 0;0 -1')
           
    i=0
    while i<len(ham_array):
        term=1
        pauli_term=ham_array[i]
        
        for j in range(len(pauli_term)):
                         
            if pauli_term[j]=='I':
                         term=np.kron(term,identity)
            elif pauli_term[j]=='X':
                         term=np.kron(term,sx)
            elif pauli_term[j]=='Y':
                         term=np.kron(term,sy)
            elif pauli_term[j]=='Z':
                         term=np.kron(term,sz)
           
        hamiltonian=np.add(hamiltonian,float(ham_array[i+1])*term)
        i+=2
                         
    return hamiltonian 
        

def SPSA_optimization(obj_fun,initial_theta,SPSA_parameters,max_trials):
    
    # Minimizes obj_fun(controls) 
    
    theta_plus=[]
    theta_minus=[]
    theta=[]
    cost_plus=[]
    cost_minus=[]
    
    theta.append(initial_theta)
    
    for k in range(max_trials):
        
       
        #SPSA Paramaters
        a_spsa = float(SPSA_parameters[0])/np.power(k+1+SPSA_parameters[4], SPSA_parameters[2])
        c_spsa = float(SPSA_parameters[1])/np.power(k+1, SPSA_parameters[3])
        Delta=2*np.random.randint(2,size=np.shape(initial_theta)[0]) - 1
     

               
        theta_plus.append(theta[k]+c_spsa*Delta)
        theta_minus.append(theta[k]-c_spsa*Delta)
        
        cost_plus.append(obj_fun(theta_plus[k])[0])
        cost_minus.append(obj_fun(theta_minus[k])[0])
        
        print(('Energy at theta_plus for step # '+str(k)))
        print(cost_plus[k])
        print(('Energy at theta_minus for step # '+str(k)))
        print(cost_minus[k])
        g_spsa=(cost_plus[k]-cost_minus[k])*Delta/(2.0*c_spsa)
        
        theta.append(theta[k]-a_spsa*g_spsa)
        
    cost_final=obj_fun(theta[max_trials-1])[0]
    
    print('Final Energy is\n'+str(cost_final))
    
    return cost_final , cost_plus , cost_minus , theta_plus, theta_minus 
        
        
