"""Wigner function calculator example driver.

Implements the Wigner function introduced in section 
IV. A Wigner Function For Tensor Products of Spins
in reference [1]

Authors:
	* R.P. Rundle [1]
	* M.J. Everitt (m.j.everitt@physics.org) [1]
	* ...

Affiliations:
	1. Quantum Systems Engineering Research Group
	   Loughborough University, Leicestershire LE11 3TU, United Kingdom

Todo:
    * propper documentation
    * more examples
    * use jobs rather than api

Refernces:

[1] Quantum state reconstruction made easy: a direct method for tomography
    R.P. Rundle, Todd Tilma, J. H. Samson, M. J. Everitt
    https://arxiv.org/abs/1605.08922

[2] Wigner Functions for Arbitrary Quantum Systems
    T Tilma, MJ Everitt, JH Samson, WJ Munro, K Nemoto
    Physical review letters 117 (18), 180401
    https://doi.org/10.1103/PhysRevLett.117.180401
    https://arxiv.org/abs/1601.07772

License: 
Apache License Version 2.0, January 2004
http://www.apache.org/licenses/

Note: We would normally use GPL but have chosen Apache to fit with the 
IBM Quantum Experience qiskit-sdk-py.

"""

##
#  Functions that will form the library 
#

import math
import numpy as np
import matplotlib.pyplot as plt

from IBMQuantumExperience import IBMQuantumExperience

def generate_instructions_for_IBM_to_perfom_correct_rotations_and_mesuremnts(theta,phi,qubits):
    returnString=""
    for i in range(len(theta)):
        returnString = returnString + 'u3('+theta[i]+',0,'+phi[i]+') q[' + qubits[i]+ '];'
    for j in range(len(theta)):
        returnString = returnString + '\nmeasure q['+ qubits[j]+'] -> c['+str(j)+'];'
    returnString = returnString +'\n'
    return returnString

def tensor_product_parity(nQubits):
    """Calculates the tensor product parity operator for a given number of qubits"""
    P = [0.5+0.5*math.sqrt(3),0.5-0.5*math.sqrt(3)]
    par = 1
    for i in range(nQubits):
        par = np.kron(par,P)
    return par


def wigner_function(ibm_setup,theta,phi,qubits):
    """Calulates the wigner fucntion at one point in phase space.
    
        Keyword arguments:
        string -- the base string that is the start of the quantum code.
        theta -- the theta roations for each qubit
        phi -- the phi roations for each qubit
        qubits -- array containing a the actual qubits used in the calculations - assumes assnding order
        """    
    message = "As arguments specify theta, phi and index of each qubit they must be of the same length."
    assert len(theta)==len(phi), message
    assert len(theta)==len(qubits), message
    dimension=2**len(qubits)
    # setup and run code on IBM and retive values
    runString = ibm_setup['baseString'] + generate_instructions_for_IBM_to_perfom_correct_rotations_and_mesuremnts(theta,phi,qubits)
    ibm_setup['api'].run_experiment(runString, ibm_setup['device'], ibm_setup['shots'], timeout=90)
    lastCodes =ibm_setup['api'].get_last_codes()
    
    datas = lastCodes[1]['executions'][0]['result']['data']['p']
    labels = datas['labels']
    results = datas['values']

    # locally compute the wigner function from IBM output
    parity = tensor_product_parity(len(theta))
    x = [0]*dimension
    elements = len(results)
    norm = 0.0
    for i in range(elements):
        y = labels[i]
        x[int(y, 2)] = float(results[i])
        norm = norm + float(results[i])
    W = 0.0
    for i in range(dimension):
        W = W+(x[i]/norm)*parity[i]
    return W


##
#	Do some setup including commands for state preparation (currently |00>+|11>)
#

token="PUT A VALID TOKEN HERE"

api=IBMQuantumExperience(token)
###baseString includes creation of a Bell state

setup = { "api"       : api,
          "config"    : {'url': 'https://quantumexperience.ng.bluemix.net/api'},
          "device"    : 'simulator',
          "shots"     : 1024,
          "baseString": 'OPENQASM 2.0;\n\ninclude "qelib1.inc";\nqreg q[5];\ncreg c[5];\nh q[0];\ncx q[0],q[2];'
         }

##
#	Calculate and plot an equatorial section of the equal angle slice of the Wigner function
#

number_output_points=20
W=[0]*number_output_points
phis=[0]*number_output_points

for phiindex in range(0, number_output_points):
    phi_string = str(phiindex)+'*2*pi/'+str(number_output_points)
    theta = ['pi/2','pi/2']
    phi = [phi_string,phi_string]
    qubits = ['0','2']
    print(phi_string)
    
    W[phiindex] = wigner_function(setup,theta,phi,qubits)

print(W)
plt.plot(W)
plt.show()
