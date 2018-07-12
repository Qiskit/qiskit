"""
To generate a circuit for counterfeit-coin finding
algorithm using 15 coins and the false coin is the third coin,
type the following.

python cc_gen.py -c 15 -f 3

@author Raymond Harry Rudy rudyhar@jp.ibm.com
"""
import random
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class CounterfeitCoin:
    """
    CC Generator
    """
    def __init__(self, seed):
        self.name = "cc"
        self.seed = seed

    @classmethod
    def generate_false(cls, ncoins):
        """
            generate a random index of false coin (counting from zero)
        """
        return random.randint(0, ncoins-1)

    @classmethod
    def gen_cc_main(cls, ncoins, index_of_false_coin):
        """
            generate a circuit of the counterfeit coin problem
        """
        # using the last qubit for storing the oracle's answer
        nqubits = ncoins + 1
        # Creating registers
        # qubits for querying coins and storing the balance result
        q_r = QuantumRegister(nqubits)
        # for recording the measurement on qr
        c_r = ClassicalRegister(nqubits)
        cccircuit = QuantumCircuit(q_r, c_r)

        # Apply Hadamard gates to the first ncoins quantum register
        # create uniform superposition
        for i in range(ncoins):
            cccircuit.h(q_r[i])

        # check if there are even number of coins placed on the pan
        for i in range(ncoins):
            cccircuit.cx(q_r[i], q_r[ncoins])

        # perform intermediate measurement to check if the last qubit is zero
        cccircuit.measure(q_r[ncoins], c_r[ncoins])

        # proceed to query the quantum beam balance if cr is zero
        cccircuit.x(q_r[ncoins]).c_if(c_r, 0)
        cccircuit.h(q_r[ncoins]).c_if(c_r, 0)

        # we rewind the computation when cr[N] is not zero
        for i in range(ncoins):
            cccircuit.h(q_r[i]).c_if(c_r, 2**ncoins)

        # apply barrier for marking the beginning of the oracle
        cccircuit.barrier()

        cccircuit.cx(q_r[index_of_false_coin], q_r[ncoins]).c_if(c_r, 0)

        # apply barrier for marking the end of the oracle
        cccircuit.barrier()

        # apply Hadamard gates to the first ncoins qubits
        for i in range(ncoins):
            cccircuit.h(q_r[i]).c_if(c_r, 0)

        # measure qr and store the result to cr
        for i in range(ncoins):
            cccircuit.measure(q_r[i], c_r[i])

        return cccircuit

    def gen_application(self, app_arg):
        """
        generate application
        """
        random.seed(self.seed)

        qubits = app_arg["qubit"]
        qubits = qubits - 1
        falseindex = None
        if falseindex is None:
            falseindex = self.generate_false(qubits)
        circ = self.gen_cc_main(qubits, falseindex)

        return circ
