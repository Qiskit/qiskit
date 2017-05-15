"""
Helper methods for single qubit state tomography.

Author: Andrew Cross
"""
import copy


def expectation_Z(data, j):
    """Compute the expectation value of Z for the jth qubit.

    The data takes the form {"bitstring": count,...}
    """
    shots = sum(data.values())
    total = 0.0
    for k, v in data.items():
        total += (-1)**int(k[len(k) - j - 1]) * float(v) / float(shots)
    return total


# Measurement instruction
meas_str = "measure %s[%d] -> %s[%d];\n"


def generate_tomo1_circuits(baseQASM, tomo_qubits,
                            input_qreg='q', input_creg='c'):
    """Generate a collection of circuits to implement 1Q state tomography.

    baseQASM = input QASM source file with one input_qreg and one input_creg
    of the same size of the qreg, no measurements, and includes "qelib1.inc"
    on line 2.
    tomo_qubits = list of indices of input_qreg for 1Q state tomography
    input_qreg = string containing qreg name
    input_creg = string containing creg name
    """
    job = []
    for post_rotation in [['h'], ['sdg', 'h'], []]:
        qasm = {'qasm': copy.copy(baseQASM)}
        for q in tomo_qubits:
            for g in post_rotation:
                qasm['qasm'] += "%s %s[%d];\n" % (g, input_qreg, q)
            qasm['qasm'] += meas_str % (input_qreg, q, input_creg, q)
        job.append(qasm)
    return job
