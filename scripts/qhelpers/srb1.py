"""
Helper methods for making single qubit randomized benchmarking experiments.

These are standard randomized benchmarking sequences over the single qubit
Clifford group. The multiplication and inversion operations are
implemented by a lookup in a precomputed table. Each Clifford gate has a
corresponding realization as either a u1, u2, or u3 gate on the
Quantum Experience. We chose the realization to minimize the number of
physical pulses, i.e. we prefer u1 to u2 and u2 to u3.

Author: Andrew Cross
"""
import random
import copy
import qhelpers.misc as misc


def prob_0(data, j):
    """Compute Prob(0) for the jth qubit.

    The data takes the form {"bitstring": counts,...}
    """
    shots = sum(data.values())
    total = 0.0
    for k, v in data.items():
        total += (1-int(k[len(k)-j-1]))*float(v)/float(shots)
    return total


def survival_prob(m, a, b, alpha):
    """Compute a*alpha**m+b for array m.

    a is positive real
    b is real
    alpha is real in [0,1]
    """
    return list(map(lambda x: a*alpha**x+b, m))


def process_results(results, total_sequences, total_length, step, qubit):
    """Process the raw results into survival probabilities and average them.

    results is a list of results returned from getJob
    total_sequences = total number of RB sequences to generate
    total_length = total number of Clifford gates in each sequence
    step = increment to number of Clifford gates for subsequences
    qubit = one of 0 through 4 for the 5 qubit QE device

    Return a 3-tuple (xdata, ydatas, yavg). The xdata is a list of
    subsequence lengths. The ydatas is a list of total_sequences lists
    containing the survival probabilities at each subsequence length.
    The yavg contains the mean survival probability over all sequences
    at each subsequence length.
    """
    xdata = [seq_len for seq_len in range(step, total_length + step, step)]
    ydatas = []
    # For each sequence, compute the subsequence results
    for seq_num in range(total_sequences):
        # Append a new y data array
        ydatas.append([])
        # For each subsequence length, compute survival probability
        for seq_len in range(step, total_length + step, step):
            j = int((seq_len - step)/step)
            data = misc.get_data(results[seq_num], j)
            ydatas[-1].append(prob_0(data, qubit))
    # Compute the mean survival probability over all sequences
    yavg = []
    for j in range(len(xdata)):
        yavg.append(sum([r[j] for r in ydatas])/total_sequences)
    return (xdata, ydatas, yavg)


# The 24 elements of the single qubit Clifford group
clifford_gates = ["id",                      # identity
                  "u2(0,pi)",                # h
                  "u1(pi/2)",                # s
                  "u2(pi/2,3*pi/2)",         # s.h.s
                  "u2(-pi/2,-pi)",           # h.s.h.s
                  "u2(2*pi,pi/2)",           # s.h.s.h
                  "u3(-pi,0,-pi)",           # x
                  "u2(0,2*pi)",              # x.h
                  "u3(-pi,0,-pi/2)",         # x.s
                  "u2(-pi/2,-3*pi/2)",       # x.s.h.s
                  "u2(-3*pi/2,2*pi)",        # x.h.s.h.s
                  "u2(-2*pi,3*pi/2)",        # x.s.h.s.h
                  "u3(-pi,0,2*pi)",          # y
                  "u2(pi,2*pi)",             # y.h
                  "u3(-pi,0,-3*pi/2)",       # y.s
                  "u2(pi/2,-3*pi/2)",        # y.s.h.s
                  "u2(-pi/2,2*pi)",          # y.h.s.h.s
                  "u2(pi,-pi/2)",            # y.s.h.s.h
                  "u1(pi)",                  # z
                  "u2(pi,pi)",               # z.h
                  "u1(3*pi/2)",              # z.s
                  "u2(-pi/2,-pi/2)",         # z.s.h.s
                  "u2(pi/2,-pi)",            # z.h.s.h.s
                  "u2(pi,-3*pi/2)"]          # z.s.h.s.h

# Index of the product clifford_gates[i]*clifford_gates[j] is multab[i][j]
multab = [
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
          14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
         [1, 0, 11, 4, 3, 20, 19, 18, 17, 22, 21, 2, 13,
          12, 23, 16, 15, 8, 7, 6, 5, 10, 9, 14],
         [2, 22, 18, 17, 1, 15, 14, 10, 6, 5, 13, 3, 8,
          16, 12, 23, 7, 21, 20, 4, 0, 11, 19, 9],
         [3, 5, 10, 6, 2, 7, 9, 11, 4, 0, 8, 1, 21, 23,
          16, 12, 20, 13, 15, 17, 22, 18, 14, 19],
         [4, 20, 21, 19, 11, 18, 22, 2, 3, 1, 17, 0, 10,
          14, 15, 13, 5, 12, 16, 8, 9, 7, 23, 6],
         [5, 3, 1, 2, 6, 22, 17, 15, 13, 14, 18, 10, 23,
          21, 19, 20, 12, 4, 11, 9, 7, 8, 0, 16],
         [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 18, 19,
          20, 21, 22, 23, 12, 13, 14, 15, 16, 17],
         [7, 6, 5, 10, 9, 14, 13, 12, 23, 16, 15, 8, 19,
          18, 17, 22, 21, 2, 1, 0, 11, 4, 3, 20],
         [8, 16, 12, 23, 7, 21, 20, 4, 0, 11, 19, 9, 2,
          22, 18, 17, 1, 15, 14, 10, 6, 5, 13, 3],
         [9, 11, 4, 0, 8, 1, 3, 5, 10, 6, 2, 7, 15, 17,
          22, 18, 14, 19, 21, 23, 16, 12, 20, 13],
         [10, 14, 15, 13, 5, 12, 16, 8, 9, 7, 23, 6, 4,
          20, 21, 19, 11, 18, 22, 2, 3, 1, 17, 0],
         [11, 9, 7, 8, 0, 16, 23, 21, 19, 20, 12, 4, 17,
          15, 13, 14, 18, 10, 5, 3, 1, 2, 6, 22],
         [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
          23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         [13, 12, 23, 16, 15, 8, 7, 6, 5, 10, 9, 14, 1,
          0, 11, 4, 3, 20, 19, 18, 17, 22, 21, 2],
         [14, 10, 6, 5, 13, 3, 2, 22, 18, 17, 1, 15, 20,
          4, 0, 11, 19, 9, 8, 16, 12, 23, 7, 21],
         [15, 17, 22, 18, 14, 19, 21, 23, 16, 12, 20, 13,
          9, 11, 4, 0, 8, 1, 3, 5, 10, 6, 2, 7],
         [16, 8, 9, 7, 23, 6, 10, 14, 15, 13, 5, 12, 22,
          2, 3, 1, 17, 0, 4, 20, 21, 19, 11, 18],
         [17, 15, 13, 14, 18, 10, 5, 3, 1, 2, 6, 22, 11,
          9, 7, 8, 0, 16, 23, 21, 19, 20, 12, 4],
         [18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17,
          6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5],
         [19, 18, 17, 22, 21, 2, 1, 0, 11, 4, 3, 20, 7, 6,
          5, 10, 9, 14, 13, 12, 23, 16, 15, 8],
         [20, 4, 0, 11, 19, 9, 8, 16, 12, 23, 7, 21, 14, 10,
          6, 5, 13, 3, 2, 22, 18, 17, 1, 15],
         [21, 23, 16, 12, 20, 13, 15, 17, 22, 18, 14, 19, 3,
          5, 10, 6, 2, 7, 9, 11, 4, 0, 8, 1],
         [22, 2, 3, 1, 17, 0, 4, 20, 21, 19, 11, 18, 16, 8,
          9, 7, 23, 6, 10, 14, 15, 13, 5, 12],
         [23, 21, 19, 20, 12, 4, 11, 9, 7, 8, 0, 16, 5, 3,
          1, 2, 6, 22, 17, 15, 13, 14, 18, 10]
         ]


# Index of the inverse of clifford_gates[i] is invtab[i]
invtab = [0, 1, 20, 9, 11, 22, 6, 19, 8, 3, 23, 4, 12, 13,
          14, 15, 17, 16, 18, 7, 2, 21, 5, 10]


# Header for an RB experiment
rb_header = """IBMQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
"""


def generate_srb1_sequences(total_sequences, total_length, step, qubit):
    """Generate a collection of randomized benchmarking (RB) sequences.

    This method is limited to single qubit standard Clifford benchmarking.

    total_sequences = total number of RB sequences to generate
    total_length = total number of Clifford gates in each sequence
    step = increment to number of Clifford gates for subsequences
    qubit = one of 0 through 4 for the 5 qubit QE device

    Returns a list of jobs. Each job is a list of dictionaries containing
    QASM source under the key 'qasm'. Each job is one RB sequence presented
    as a list of subsequences of increasing length (Clifford gates).
    Subsequence lengths belong to range(step, total_length + step, step)
    plus one inverting gate.
    """
    # Check arguments
    if total_length % step != 0:
        print("total_length should be a multiple of step")
        return []
    # Generate random sequences
    indices = []
    for i in range(total_sequences):
        indices.append([random.randint(0, 23) for j in range(total_length)])

    # Compose the QASM source for each sequence
    jobs = []
    for seq_num in range(total_sequences):
        source = []
        for seq_len in range(step, total_length + step, step):
            qasm = {'qasm': copy.copy(rb_header)}
            clifprod = 0
            for j in range(seq_len):
                qasm['qasm'] += "%s q[%d];\n" % (
                                        clifford_gates[indices[seq_num][j]],
                                        qubit)
                clifprod = multab[indices[seq_num][j]][clifprod]
            qasm['qasm'] += "%s q[%d];\n" % (clifford_gates[invtab[clifprod]],
                                             qubit)
            qasm['qasm'] += "measure q[%d] -> c[%d];" % (qubit, qubit)
            source.append(qasm)
        jobs.append(source)
    return jobs
