"""
=============================================
Quickly simulating circuits (:mod:`qiskit.providers.quick_simulate`)
=============================================
.. currentmodule:: qiskit.providers.quick_simulate
.. autofunction:: quick_simulate
"""


from qiskit.providers.basicaer import BasicAer
from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.execute import execute 
from qiskit.exceptions import QiskitError


def quick_simulate(circuit, shots=1024, x="0", verbose=False):
	"""simulates circuit with given input

	Args:
	    circuit (QuantumCircuit): Circuit to be simulated. Currently no more than registers supported
	    shots (int, optional): number of shots to simulate. Default 1024
	    x (str, optional): input string. eg "1011" would apply an X gate to the first, third and fourth qubits. Default "0"
	    verbose (bool, optional): prints extra output

	Returns:
	    dict[str:int]: results.get_counts()
	"""
	names = []
	regs = []
	for q in circuit.qubits:
		name = q.register.name
		size = len(q.register)
		if name not in names:
			names.append(name)
			regs.append(size)

	if verbose: print(names, regs)

	#assuming that we only have 2: control + anciallary
	qra = QuantumRegister(regs[0], name=names[0])
	if len(regs) > 2:
		raise QiskitError("Not yet implemented for more than 2 registers")
	elif len(regs) == 2:
		qran = QuantumRegister(regs[1], name=names[1])
		qa = QuantumCircuit(qra,qran)
	else:
		qa = QuantumCircuit(qra)

	if len(x) != sum(regs): x += "0" * (sum(regs) - len(x))
	if verbose: print(x)
	for bit in range(len(x)):
		if verbose: print(x[bit], type(x[bit]))
		if x[bit] != "0":
			qa.x(bit)
	qa.barrier()

	qa.extend(circuit)

	if verbose:
		print(qa)

	backend = BasicAer.get_backend('qasm_simulator')
	results = execute(qa, backend=backend, shots=shots).result()
	answer = results.get_counts()
	return answer
