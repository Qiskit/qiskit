from functools import reduce
from qiskit.opflow import I, X, Y, Z
from qiskit.algorithms import QAOA
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import AdaptQAOA


QISKIT_DICT = {"I": I, "X": X, "Y": Y, "Z": Z}

def string_to_qiskit(qstring):
    if is_all_same(qstring):
        # case where its all X's or Y's
        gate = qstring[0]
        list_string = [
            i * "I" + gate + (len(qstring) - i - 1) * "I" for i in range(len(qstring))]
        return sum([reduce(lambda a, b: a ^ b, [QISKIT_DICT[char.upper()] for char in x]) for x in list_string])

    return reduce(lambda a, b: a ^ b, [QISKIT_DICT[char.upper()] for char in qstring])
def is_all_same(items):
    return all(x == items[0] for x in items)


mixer_list = ["XXIII","XIIX","IXXII"]
cost_op = string_to_qiskit("ZZZZZ")
mixer_list = [string_to_qiskit(x) for x in mixer_list]

quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

'''
adapt = AdaptQAOA(mixer_pool_type='singular', max_reps=5, quantum_instance=quantum_instance)
#adapt.optimal_mixer_list = mixer_list
cme = adapt.compute_minimum_eigenvalue(cost_op)
print(cme)
'''

adaptqaoa = AdaptQAOA(max_reps=6, quantum_instance=quantum_instance, mixer_pool_type="singular")
out = adaptqaoa.run_adapt(cost_op)
print("Adapt result: ", out)
print(adaptqaoa.get_optimal_circuit().draw())

# qaoa = QAOA(reps=5, quantum_instance=quantum_instance)
# out = qaoa.compute_minimum_eigenvalue(cost_op)
# print(out)
# print(qaoa.get_optimal_circuit().draw())
