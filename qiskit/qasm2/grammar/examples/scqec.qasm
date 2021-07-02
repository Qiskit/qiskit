/*
 * Surface code quantum memory.
 *
 * Estimate the failure probability as a function
 * of parameters at the top of the file.
 */
OPENQASM 3;
include "stdgates.inc";

const d = 3;         // code distance
const m = 10;        // number of syndrome measurement cycles
const shots = 1000;  // number of samples
const n = d^2;       // number of code qubits

uint[32] failures;  // number of observed failures

kernel zfirst(creg[n - 1], int[32], int[32]);
kernel send(creg[n -1 ], int[32], int[32], int[32]);
kernel zlast(creg[n], int[32], int[32]) -> bit;

qubit data[n];  // code qubits
qubit ancilla[n - 1];  // syndrome qubits
/*
 * Ancilla are addressed in a (d-1) by (d-1) square array
 * followed by 4 length (d-1)/2 arrays for the top,
 * bottom, left, and right boundaries.
 */

bit layer[n - 1];  // syndrome outcomes in a cycle
bit data_outcomes[n];  // data outcomes at the end
bit outcome;  // logical outcome

/* Declare a sub-circuit for Hadamard gates on ancillas
 */
def hadamard_layer qubit[n-1]:ancilla {
  // Hadamards in the bulk
  for row in [0: d-2] {
    for col in [0: d-2] {
      bit sum[32] = bit[32](row + col);
      if(sum[0] == 1)
        h ancilla[row * (d - 1) + col];
    }
  }
  // Hadamards on the left and right boundaries
  for i in [0: d - 2] {
    h ancilla[(d - 1)^2 + (d - 1) + i];
  }
}

/* Declare a sub-circuit for a syndrome cycle.
 */
def cycle qubit[n]:data, qubit[n-1]:ancilla -> bit[n-1] {
  reset ancilla;
  hadamard_layer ancilla;

  // First round of CNOTs in the bulk
  for row in [0: d - 2] {
    for col in [0:d - 2] {
      bit sum[32] = bit[32](row + col);
      if(sum[0] == 0)
        cx data[row * d + col], ancilla[row * (d - 1) + col];
      if(sum[0] == 1) {
        cx ancilla[row * (d - 1) + col], data[row * d + col];
      }
    }
  }
  // First round of CNOTs on the bottom boundary
  for i in [0: (d - 3) / 2] {
    cx data[d * (d - 1) + 2 * i], ancilla[(d - 1) ^ 2 + ( d- 1) / 2 + i];
  }
  // First round of CNOTs on the right boundary
  for i in [0: (d - 3) / 2] {
    cx ancilla[(d - 1) ^ 2 + 3 * (d - 1) / 2 + i], data[2 * d - 1 + 2 * d * i];
  }
  // Remaining rounds of CNOTs, go here ...

  hadamard_layer ancilla;
  return measure ancilla;
}

// Loop over shots
for shot in [1: shots] {

  // Initialize
  reset data;
  layer = cycle data, ancilla;
  zfirst(layer, shot, d);

  // m cycles of syndrome measurement
  for i in [1: m] {
    layer = cycle data, ancilla;
    send(layer, shot, i, d);
  }

  // Measure
  data_outcomes = measure data;

  outcome = zlast(data_outcomes, shot, d);
  failures += int[1](outcome);
}

/* The ratio of "failures" to "shots" is our result.
 * The data can be logged by the external functions too.
 */
