#Author: Justin Roberts
#Date: 22 March 2018
#A helper script for running your IBM Quantum Experience scores
#The following implements a quantum score for examining 16 qubit entanglement on ibmqx5

print("\nconnecting....")

#import matplotlib.pyplot as plt
#from qiskit.tools.visualization import plot_histogram
#You can uncomment the above if you want to try and plot the histogram but I wouldn't if I were you :)

from qiskit import QuantumProgram
import Qconfig,time

def QuantumScript():
	
	#The following circuit attempts to entangle all qubits on the ibmqx5 processor
	#However, you can just copy and paste your other circuits into here instead if you so wish

	global qp
	Q_SPECS = {
   	 'circuits': [{
        	'name': 'Circuit',
        	'quantum_registers': [{
        	    'name': 'qr',
        	    'size': 16
        	}],
        	'classical_registers': [{
        	    'name': 'cr',
        	    'size': 16
        	}]}],
	}

	qp = QuantumProgram(specs=Q_SPECS)
	qp.set_api(Qconfig.APItoken, Qconfig.config['url'])

	circuit = qp.get_circuit('Circuit')
	quantum_r = qp.get_quantum_register('qr')
	classical_r = qp.get_classical_register('cr')

	circuit.h(quantum_r[1])
	circuit.h(quantum_r[2])
	circuit.h(quantum_r[5])
	circuit.h(quantum_r[6])
	circuit.h(quantum_r[9])
	circuit.h(quantum_r[11])
	circuit.h(quantum_r[12])
	circuit.h(quantum_r[15])

	circuit.cx(quantum_r[1],quantum_r[0])
	circuit.cx(quantum_r[2],quantum_r[3])
	circuit.cx(quantum_r[5],quantum_r[4])
	circuit.cx(quantum_r[6],quantum_r[7])
	circuit.cx(quantum_r[9],quantum_r[8])
	circuit.cx(quantum_r[11],quantum_r[10])
	circuit.cx(quantum_r[12],quantum_r[13])
	circuit.cx(quantum_r[15],quantum_r[14])

	circuit.h(quantum_r[2])
	circuit.h(quantum_r[3])
	circuit.h(quantum_r[5])
	circuit.h(quantum_r[8])
	circuit.h(quantum_r[11])
	circuit.h(quantum_r[13])

	circuit.cx(quantum_r[1],quantum_r[2])
	circuit.cx(quantum_r[3],quantum_r[4])
	circuit.cx(quantum_r[6],quantum_r[5])
	circuit.cx(quantum_r[8],quantum_r[7])
	circuit.cx(quantum_r[9],quantum_r[10])
	circuit.cx(quantum_r[12],quantum_r[11])
	circuit.cx(quantum_r[13],quantum_r[14])
	circuit.cx(quantum_r[15],quantum_r[14])

	circuit.cx(quantum_r[15],quantum_r[0])

	circuit.h(quantum_r[0])
	circuit.h(quantum_r[2])
	circuit.h(quantum_r[4])
	circuit.h(quantum_r[5])
	circuit.h(quantum_r[7])
	circuit.h(quantum_r[10])
	circuit.h(quantum_r[11])
	circuit.h(quantum_r[14])

	circuit.barrier(quantum_r)

	circuit.measure(quantum_r[0], classical_r[0])
	circuit.measure(quantum_r[1], classical_r[1])
	circuit.measure(quantum_r[2], classical_r[2])
	circuit.measure(quantum_r[3], classical_r[3])
	circuit.measure(quantum_r[4], classical_r[4])
	circuit.measure(quantum_r[5], classical_r[5])
	circuit.measure(quantum_r[6], classical_r[6])
	circuit.measure(quantum_r[7], classical_r[7])
	circuit.measure(quantum_r[8], classical_r[8])
	circuit.measure(quantum_r[9], classical_r[9])
	circuit.measure(quantum_r[10], classical_r[10])
	circuit.measure(quantum_r[11], classical_r[11])
	circuit.measure(quantum_r[12], classical_r[12])
	circuit.measure(quantum_r[13], classical_r[13])
	circuit.measure(quantum_r[14], classical_r[14])
	circuit.measure(quantum_r[15], classical_r[15])


def BackendData():
	
	#Get all the backend availabilty and config data etc
	global bk
	global shots
	global max_cred

	print("\nAvailable backends:\n")
	
	for i in range(len(qp.available_backends())):
		print(i,qp.available_backends()[i])
	
	bki=input("\nChoose backend (enter index no.): ")
	bk=qp.available_backends()[int(bki)]
	bk_status=input("\nGet current backend status (y/n): ")
	
	if bk_status=="y":
		try:
			print('\n',qp.get_backend_status(bk))
		except:
			LookupError
			print("\nNone")
	
	bk_config=input("\nGet backend configuration (y/n): ")
	
	if bk_config=="y":
		try:
			print('\n',qp.get_backend_configuration(bk))
		except:
			LookupError
			print("\nNone")
		
	bk_calib=input("\nGet backend calibration data: (y/n): ")
	
	if bk_calib=="y":
		try:
			print('\n',qp.get_backend_calibration(bk))
		except:
			LookupError
			print("\nNone")

	bk_params=input("\nGet backend parameters (y/n): ")
	
	if bk_params=="y":
		try:
			print('\n',qp.get_backend_parameters(bk))
		except:
			LookupError
			print("\nNone")

	qasm_source=input("\nPrint qasm source y/n: ")
	
	if qasm_source=="y":
		QASM_source = qp.get_qasm('Circuit')
		print('\n',QASM_source)
	
	shots=input("\nshots (1-8192): Default=1024: ")
	
	if shots=='':
		shots=str(1024)
	
	max_cred=input("\nmaximum credits to use. Default=3: ")
	
	if max_cred=='':
		max_cred=str(3)
		print("\n.....maximum credits set to 3")
		time.sleep(1)

QuantumScript()
BackendData()

warn=input("\nYou are about to run this circuit on "+str(bk)+".\nAre you sure (y/n): ")

if warn=="n" or '':
	BackendData()
	warn=input("\nYou are about to run this circuit on "+str(bk)+".\nAre you sure (y/n): ")
else:
	pass 

circuits = ['Circuit']
print("\nProcessing....\n")
result = qp.execute(circuits, bk, shots=int(shots),max_credits=int(max_cred),wait=10, timeout=1000)
res1=result.get_counts('Circuit')

#res2=result.get_data('Circuit')
#print(res1)
#print(res2)

#comment out the below for-loop and uncomment the above if you want to return the results as a dictionary
for key,val in res1.items():
	print(key,int(key,2),val)

#plot_histogram(res1)

compiled_qasm=input("\nGet compiled qasm y/n:")
if compiled_qasm=="y":
	ran_qasm = result.get_ran_qasm('Circuit')
	print('\n',ran_qasm)
else:
	print ("\nDone...\n")
