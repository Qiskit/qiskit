"""
To generate a Bernstein-Vazirani algorithm using 5 qubits, type the following.

python bv_gen.py -q 5 -o bv5
The resulting circuit is stored at bv5.qasm and its drawing at bv5.tex.

For more details, run the above command with -h or --help argument.

@author Raymond Harry Rudy rudyhar@jp.ibm.com
"""
import random
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class BersteinVazirani:
    """
    Berstein-Vazirani Generator
    """
    def __init__(self, seed):
        self.name = "bv"
        self.seed = seed

    @classmethod
    def generate_astring(cls, nqubits, prob=1.0):
        """
            generate a random binary string as a hidden bit string
        """
        answer = []
        for _ in range(nqubits):
            if random.random() <= prob:
                answer.append("1")
            else:
                answer.append("0")

        return "".join(answer)

    @classmethod
    def bin2int(cls, alist):
        """
            convert a binary string into integer
        """
        answer = 0
        temp = alist
        temp.reverse()
        for i in range(temp):
            answer += 2**int(temp[i])
        temp.reverse()
        return answer

    @classmethod
    def check_astring(cls, astring, nqubits):
        """
            check the validity of string
        """
        if len(astring) > nqubits:
            raise Exception("The length of the hidden string is \
                            longer than the number of qubits")
        else:
            for i in astring:
                if i != "0" and i != "1":
                    raise Exception("Found nonbinary string at "+astring)
        return True

    @classmethod
    def gen_bv_main(cls, nqubits, hiddenstring):
        """
            generate a circuit of the Bernstein-Vazirani algorithm
        """
        # Creating registers
        # qubits for querying the oracle and finding the hidden integer
        q_r = QuantumRegister(nqubits)
        # for recording the measurement on qr
        c_r = ClassicalRegister(nqubits-1)

        bvcircuit = QuantumCircuit(q_r, c_r)

        # Apply Hadamard gates to the first
        # (nQubits - 1) before querying the oracle
        for i in range(nqubits-1):
            bvcircuit.h(q_r[i])

        # Apply 1 and Hadamard gate to the last qubit
        # for storing the oracle's answer
        bvcircuit.x(q_r[nqubits-1])
        bvcircuit.h(q_r[nqubits-1])

        # Apply barrier so that it is not optimized by the compiler
        bvcircuit.barrier()

        # Apply the inner-product oracle
        hiddenstring = hiddenstring[::-1]
        for index, element in enumerate(hiddenstring):
            if element == "1":
                bvcircuit.cx(q_r[index], q_r[nqubits-1])
        hiddenstring = hiddenstring[::-1]
        # Apply barrier
        bvcircuit.barrier()

        # Apply Hadamard gates after querying the oracle
        for i in range(nqubits-1):
            bvcircuit.h(q_r[i])

        # Measurement
        for i in range(nqubits-1):
            bvcircuit.measure(q_r[i], c_r[i])

        return bvcircuit

    def gen_application(self, app_arg):
        """
        generate application
        """
        random.seed(self.seed)

        qubits = app_arg["qubit"]
        prob = 1.0
        hiddenstring = None

        if hiddenstring is None:
            hiddenstring = self.generate_astring(qubits-1, prob)

        qcirc = self.gen_bv_main(qubits, hiddenstring)
        return qcirc
