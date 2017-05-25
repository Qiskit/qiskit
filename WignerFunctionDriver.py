"""Wigner function calculator example driver.
+
+Implements the Wigner function introduced in section 
+IV. A Wigner Function For Tensor Products of Spins
+in reference [1]
+
+Authors:
+    * R.P. Rundle [1]
+    * M.J. Everitt (m.j.everitt@physics.org) [1]
+    * ...
+
+Affiliations:
+    1. Quantum Systems Engineering Research Group
+       Loughborough University, Leicestershire LE11 3TU, United Kingdom
+
+Todo:
+    * propper documentation
+    * more examples
+    * use jobs rather than api
+
+Refernces:
+
+[1] Quantum state reconstruction made easy: a direct method for tomography
+    R.P. Rundle, Todd Tilma, J. H. Samson, M. J. Everitt
+    https://arxiv.org/abs/1605.08922
+
+[2] Wigner Functions for Arbitrary Quantum Systems
+    T Tilma, MJ Everitt, JH Samson, WJ Munro, K Nemoto
+    Physical review letters 117 (18), 180401
+    https://doi.org/10.1103/PhysRevLett.117.180401
+    https://arxiv.org/abs/1601.07772
+
+License: 
+Apache License Version 2.0, January 2004
+http://www.apache.org/licenses/
+
+Note: We would normally use GPL but have chosen Apache to fit with the 
+IBM Quantum Experience qiskit-sdk-py.
+
+"""
##
#  Functions that will form the library 
#

import math
import numpy as np
import matplotlib.pyplot as plt
from IBMQuantumExperience  import IBMQuantumExperience
def generate_instructions_for_IBM_to_perfom_correct_rotations_and_mesurements(theta,phi,qubits):
    returnString=""
    for i in range(len(theta)):
        returnString = returnString + 'u3('+theta[i]+',0,'+phi[i]+') q[' + qubits[i]+ '];'
    for j in range(len(theta)):
        returnString = returnString + '\nmeasure q['+ qubits[j]+'] -> c['+str(j)+'];'
    returnString = returnString +'\n'
    return returnString

def generate_qasms_for_batch_runs(baseString, qubits, number_output_points, theta):
    phi        = [0]*len(qubits)
    runStrings = [0]*(number_output_points)
    for phiindex in range(0,number_output_points):
        for i in range(0,len(qubits)):
            phi_string  = str(phiindex)+'*2*pi/'+str(number_output_points)
            phi[i]      = phi_string
        runStrings[phiindex] = {'qasm': baseString + generate_instructions_for_IBM_to_perfom_correct_rotations_and_mesurements(theta,phi,qubits)}
    return runStrings

def tensor_product_parity(nQubits):
    """Calculates the tensor product parity operator for a given number of qubits"""
    P = [0.5+0.5*math.sqrt(3),0.5-0.5*math.sqrt(3)]
    par = 1
    for i in range(nQubits):
        par = np.kron(par,P)
    return par


def wigner_function(ibm_setup,runStrings,qubits,number_output_points):
    """Calulates the wigner fucntion at one point in phase space.
    
        Keyword arguments:
        string -- the base string that is the start of the quantum code.
        theta -- the theta roations for each qubit
        phi -- the phi roations for each qubit
        qubits -- array containing a the actual qubits used in the calculations - assumes assnding order
        """    
    #message = "As arguments specify theta, phi and index of each qubit they must be of the same length."
    #assert len(theta)==len(phi), message
    #assert len(theta)==len(qubits), message
    number_qubits = len(qubits)
    dimension=2**number_qubits
    # setup and run code on IBM and retive values
 
    job = ibm_setup['api'].run_job(runStrings, ibm_setup['device'], ibm_setup['shots'])
    #lastCodes =ibm_setup['api'].get_last_codes()
    #print(lastCodes)
    #print(job)
    #job_id = job[1]
    #print(job_id)
    job_id = job['id']
    print('job id' + job_id)
    lastCodes = ibm_setup['api'].get_job(job_id)
    #print(lastCodes)
    datas = [0]*number_output_points
    for i in range(0,number_output_points):
        datas[i] = lastCodes['qasms'][i]['result']['data']['counts']
    #labels = datas['labels']
    #results = datas['values']

    # locally compute the wigner function from IBM output
    parity = tensor_product_parity(len(theta))
    W = [0]*number_output_points
    for point in range(0,number_output_points):
        x = [0]*dimension
        norm = 0.0
        for i in range(dimension):
            if bin(i)[2:].zfill(number_qubits) in datas[point]:
                x[i] = float(datas[point][bin(i)[2:].zfill(number_qubits)])
                norm = norm + float(datas[point][bin(i)[2:].zfill(number_qubits)])
        for i in range(dimension):
            W[point] = W[point]+(x[i]/norm)*parity[i]
    return W


##
#    Do some setup including commands for state preparation (currently |00>+|11>)
#
token = 'token'  ### <-- put your token here
api=IBMQuantumExperience(token)
number_of_qubits = 3
###baseString includes creation of a Bell state
baseString1 = 'OPENQASM 2.0;\n\ninclude "qelib1.inc";\nqreg q['+str(number_of_qubits)+'];\ncreg c['+str(number_of_qubits)+'];'
###assuming qubit 7 is central qubit
centralQubit = 2
baseString2 = '\nh q[0];'
baseString3 = '\ncx q[0],q['+str(centralQubit)+'];'
baseString4 = '\nh q[0];'
for i in range(1,number_of_qubits):
        if i != centralQubit:
            baseString2 = baseString2 + '\nh q['+str(i)+'];'
            baseString3 = baseString3 + '\ncx q['+str(i)+'],q['+str(centralQubit)+'];'
            baseString4 = baseString4 + '\nh q['+str(i)+'];'
        else:
            baseString2 = baseString2 + '\nx q['+str(i)+'];'
            baseString4 = baseString4 + '\nh q['+str(i)+'];'
            
baseString = baseString1+baseString2+baseString3+baseString4
setup = { "api"       : api,
          "config"    : {'url': 'https://quantumexperience.ng.bluemix.net/api'},
          "device"    : 'simulator',
          "shots"     : 4000,
          "baseString": baseString
         }

###
#    Calculate and plot an equatorial section of the equal angle slice of the Wigner function
#

number_output_points=160

W            = [0]*number_output_points
equatorAngle = [0]*number_output_points
theta        = [0]*number_of_qubits
qubits       = [0]*number_of_qubits

for i in range(0,number_of_qubits):
    theta[i]  = 'pi/2'
    qubits[i] = str(i)
for i in range(0,number_output_points):
    equatorAngle[i] = i*2*math.pi/number_output_points

runStrings = generate_qasms_for_batch_runs(baseString, qubits, number_output_points, theta)

batchSize = 10

for i in range(0,(number_output_points/batchSize)):
    W[i*batchSize:(i+1)*batchSize] = wigner_function(setup,runStrings[i*batchSize:(i+1)*batchSize], qubits, batchSize)
    print(W)
    
plt.plot(equatorAngle,W)
plt.show()
