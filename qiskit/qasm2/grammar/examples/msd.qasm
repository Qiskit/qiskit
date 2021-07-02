/*
 * Magic state distillation and computation
 */
OPENQASM 3;
include "stdgates.inc";

const buffer_size = 6;  // size of magic state buffer

// Y-basis measurement
def ymeasure qubit:q -> bit {
  s q;
  h q;
  return measure q;
}

/*
 * Distillation subroutine takes 10 |H> magic states
 * and 3 scratch qubits that will be reinitialized.
 * The first two input magic states are the outputs.
 * The subroutine returns a success bit that is true
 * on success and false otherwise (see arXiv:1811.00566).
 */
def distill qubit[10]:magic, qubit[3]:scratch -> bool {
  bit temp;
  bit checks[3];
  // Encode two magic states in the [[4,2,2]] code
  reset scratch[0: 1];
  h scratch[1];
  cx scratch[1], magic[0];
  cx magic[1], scratch[0];
  cx magic[0], scratch[0];
  cx scratch[1], magic[1];
  // Body of distillation circuit
  cy magic[2], scratch[0];
  h magic[1];
  temp = ymeasure magic[2];
  if(temp == 1) { ry(-pi / 2) scratch[0]; }
  reset scratch[2];
  h scratch[2];
  cz scratch[2], scratch[0];
  cy magic[3], scratch[0];
  temp = ymeasure magic[3];
  if(temp==0) { ry(pi / 2) scratch[0]; }
  h scratch[0];
  s scratch[0];
  cy magic[4], scratch[1];
  temp = ymeasure magic[4];
  if(temp==1) { ry(-pi / 2) scratch[1]; }
  cz scratch[3], scratch[2];
  cy magic[5], scratch[1];
  temp = ymeasure magic[5];
  if(temp==0) { ry(pi / 2) scratch[1]; }
  cy scratch[0], magic[1];
  inv @ s scratch[1];
  cz scratch[0], scratch[1];
  h scratch[0];
  cy scratch[1], magic[1];
  cy magic[6], scratch[0];
  temp = ymeasure magic[6];
  if(temp == 1) { ry(-pi / 2) scratch[0]; }
  cz scratch[2], scratch[1];
  cz scratch[2], scratch[0];
  cy magic[7], scratch[0];
  temp = ymeasure magic[7];
  if(temp == 0) ry(pi / 2) scratch[0];
  cy magic[8], scratch[1];
  temp = ymeasure magic[8];
  if(temp==1) { ry(-pi / 2) scratch[1]; }
  cz scratch[2], scratch[1];
  cy magic[9], scratch[1];
  temp = ymeasure magic[9];
  if(temp == 0) { ry(pi / 2) scratch[1]; }
  h scratch[2];
  // Decode [[4,2,2]] code
  cx magic[0], scratch[0];
  cx scratch[1], magic[1];
  cx magic[1], scratch[0];
  cx scratch[1], magic[0];
  h scratch[1];
  checks = measure scratch;
  success = ~(bool(checks[0]) | bool(checks[1]) | bool(checks[2]));
  return success;
}

// Repeat level-0 distillation until success
def rus_level_0 qubit[10]:magic, qubit[3]:scratch {
  bool success;
  while(~success) {
    reset magic;
    ry(pi / 4) magic;
    success = distill magic, scratch;
  }
}

/*
 * Run two levels of 10:2 magic state distillation.
 * Both levels have two distillations running in parallel.
 * The output pairs from the first level are separated and
 * input to different second levels distillation circuits
 * because a failure in a first level circuit can lead to
 * errors on both outputs.
 * Put the requested even number of copies into the buffer.
 */
def distill_and_buffer(int[32]:num) qubit[33]:work, qubit[buffer_size]:buffer {
  int[32] index;
  bit success_0, success_1;
  let magic_lvl0 = work[0: 9];
  let magic_lvl1_0 = work[10: 19];
  let magic_lvl1_1 = work[20: 29];
  let scratch = work[30: 32];

  // Run first-level circuits until 10 successes,
  // storing the outputs for use in the second level
  for i in [0: 9] {
    rus_level_0 magic_lvl0, scratch;
    swap magic_lvl0[0], magic_lvl1_0[i];
    swap magic_lvl0[1], magic_lvl1_1[i];
  }

  // Run two second level circuits simultaneously
  success_0 = distill magic_lvl1_0, scratch_0;
  success_1 = distill magic_lvl1_1, scratch_1;

  // Move usable magic states into the buffer register
  if(success_0 && index < buffer_size) {
    swap magic_lvl1_0[0: 1], buffer[index: index + 1];
    index += 2;
  }
  if(success_1 && index < buffer_size) {
    swap magic_lvl1_1[0: 1], buffer[index: index + 1];
    index += 2;
  }
}

// Apply Ry(pi/4) to a qubit by consuming a magic state
// from the magic state buffer at address "addr"
def Ty(int[32]:addr) qubit:q, qubit[buffer_size]:buffer {
  bit outcome;
  cy buffer[addr], q;
  outcome = ymeasure buffer[addr];
  if(outcome == 1) ry(pi / 2) q;
}

qubit workspace[33];
qubit buffer[buffer_size];

qubit q[5];
bit c[5];
int[32] address;

// initialize
reset workspace;
reset buffer;
reset q;

distill_and_buffer(buffer_size) workspace, buffer;

// Consume magic states to apply some gates ...
h q[0];
cx q[0], q[1];
Ty(address) q[0], buffer;
address++;
cx q[0], q[1];
Ty(address) q[1], buffer;
address++;

// In principle each Ty gate can execute as soon as the magic
// state is available at the address in the buffer register.

// We can continue alternating state distillation and computation
// to refill and empty a circular buffer.
