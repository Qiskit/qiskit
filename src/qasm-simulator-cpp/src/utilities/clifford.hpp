/*
Copyright (c) 2017 IBM Corporation. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
 * @file    Clifford.hpp
 * @brief   Clifford class
 * @author  Sergey Bravyi <sbravyi@us.ibm.com>
 */

#ifndef _clifford_hpp_
#define _clifford_hpp_

#include <cassert>

#include "binary_vector.hpp"

// Types
using BV::BinaryVector;
using BV::uint_t;

/*******************************************************************************
 *
 * PauliOperator Class
 *
 ******************************************************************************/

struct PauliOperator {
  BinaryVector X;
  BinaryVector Z;
  bool phase;
  PauliOperator() : X(0), Z(0), phase(0){};
  explicit PauliOperator(uint64_t len) : X(len), Z(len), phase(0) {}
};

/*******************************************************************************
 *
 * Clifford Class
 *
 ******************************************************************************/

class Clifford {
public:
  // Constructors
  Clifford() = default;
  explicit Clifford(const uint64_t nqubit);

  // first n rows are destabilizers; last n rows are stabilizers
  inline PauliOperator &operator[](uint64_t j) { return table[j]; };
  inline PauliOperator &destabilizer(uint64_t n) { return table[n]; };
  inline PauliOperator &stabilizer(uint64_t n) { return table[nqubits + n]; };
  inline PauliOperator &aux() { return table[2 * nqubits]; };

  inline uint64_t size() { return nqubits; };
  inline std::vector<PauliOperator> get_table() const { return table; };

  // Apply Clifford Operations
  void CX(const uint64_t qcon, const uint64_t qtar);
  void CZ(const uint64_t q1, const uint64_t q2);
  void H(const uint64_t qubit);
  void S(const uint64_t qubit); // square root of Z
  void X(const uint64_t qubit);
  void Y(const uint64_t qubit);
  void Z(const uint64_t qubit);
  // Meas and Prep
  bool MeasZ(const uint64_t qubit, const uint64_t rand);
  bool MeasX(const uint64_t qubit, const uint64_t rand);
  bool MeasY(const uint64_t qubit, const uint64_t rand);
  void PrepZ(const uint64_t qubit, const uint64_t rand);
  void PrepX(const uint64_t qubit, const uint64_t rand);
  void PrepY(const uint64_t qubit, const uint64_t rand);
  // Meas and Prep using C rand
  inline bool MeasZ(const uint64_t qubit) { return MeasZ(qubit, rand() % 2); };
  inline bool MeasX(const uint64_t qubit) { return MeasX(qubit, rand() % 2); };
  inline bool MeasY(const uint64_t qubit) { return MeasY(qubit, rand() % 2); };
  inline void PrepZ(const uint64_t qubit) { return PrepZ(qubit, rand() % 2); };
  inline void PrepX(const uint64_t qubit) { return PrepX(qubit, rand() % 2); };
  inline void PrepY(const uint64_t qubit) { return PrepY(qubit, rand() % 2); };
  // Reset all qubits to 0
  void Reset(); // prepares all-zeros state

private:
  std::vector<PauliOperator> table;
  uint64_t nqubits;
  void rowsum(uint64_t h, uint64_t i);
  int g(bool x1, bool z1, bool x2, bool z2);
};

/*******************************************************************************
 *
 * Clifford Class Methods
 *
 ******************************************************************************/

// Constructor
Clifford::Clifford(uint64_t nq) : nqubits(nq) {
  // initial state = allzeros
  // add destabilizers
  for (uint64_t i = 0; i < nq; i++) {
    PauliOperator P(nq);
    P.X.setValue(1, i);
    P.phase = 0;
    table.push_back(P);
  }
  // add stabilizers
  for (uint64_t i = 0; i < nq; i++) {
    PauliOperator P(nq);
    P.Z.setValue(1, i);
    P.phase = 0;
    table.push_back(P);
  }
  // add auxiliary row
  PauliOperator P(nq);
  table.push_back(P);
}

// exponent of i such that P(x1,z1) P(x2,z2)=i^g P(x1+x2,z1+z2)
int Clifford::g(bool x1, bool z1, bool x2, bool z2) {
  int phase =
      (x2 * z1 * (1 + 2 * z2 + 2 * x1) - x1 * z2 * (1 + 2 * z1 + 2 * x2)) % 4;
  if (phase < 0)
    phase += 4; // now phase=0,1,2,3
  return phase;
}

void Clifford::rowsum(uint64_t h, uint64_t i) {
  int newr = 2 * table[h].phase + 2 * table[i].phase;
  for (uint64_t q = 0; q < nqubits; q++)
    newr += g(table[i].X[q], table[i].Z[q], table[h].X[q], table[h].Z[q]);
  newr %= 4;
  assert(((newr == 0) || (newr == 2)));
  table[h].phase = (newr == 2);
  table[h].X += table[i].X;
  table[h].Z += table[i].Z;
}

// Apply Clifford Operations
void Clifford::CX(const uint64_t qcon, const uint64_t qtar) {
  for (uint64_t i = 0; i < 2 * nqubits; i++)
    table[i].phase ^= table[i].X[qcon] && table[i].Z[qtar] &&
                      (table[i].X[qtar] ^ table[i].Z[qcon] ^ 1);
  for (uint64_t i = 0; i < 2 * nqubits; i++) {
    table[i].X.setValue(table[i].X[qtar] ^ table[i].X[qcon], qtar);
    table[i].Z.setValue(table[i].Z[qtar] ^ table[i].Z[qcon], qcon);
  }
}

void Clifford::CZ(const uint64_t q1, const uint64_t q2) {
  H(q2);
  CX(q1, q2);
  H(q2);
}

void Clifford::H(const uint64_t qubit) {
  for (uint64_t i = 0; i < 2 * nqubits; i++) {
    table[i].phase ^= (table[i].X[qubit] && table[i].Z[qubit]);
    // exchange X and Z
    bool b = table[i].X[qubit];
    table[i].X.setValue(table[i].Z[qubit], qubit);
    table[i].Z.setValue(b, qubit);
  }
}

void Clifford::S(const uint64_t qubit) {
  for (uint64_t i = 0; i < 2 * nqubits; i++) {
    table[i].phase ^= (table[i].X[qubit] && table[i].Z[qubit]);
    table[i].Z.setValue(table[i].Z[qubit] ^ table[i].X[qubit], qubit);
  }
}

void Clifford::X(const uint64_t qubit) {
  for (uint64_t i = 0; i < 2 * nqubits; i++)
    table[i].phase ^= table[i].Z[qubit];
}

void Clifford::Y(const uint64_t qubit) {
  for (uint64_t i = 0; i < 2 * nqubits; i++)
    table[i].phase ^= (table[i].Z[qubit] ^ table[i].X[qubit]);
}

void Clifford::Z(const uint64_t qubit) {
  for (uint64_t i = 0; i < 2 * nqubits; i++)
    table[i].phase ^= table[i].X[qubit];
}

bool Clifford::MeasZ(const uint64_t qubit, const uint64_t randint) {
  // check if there exists stabilizer anticommuting with Z_a
  // in this case the measurement outcome is random
  bool is_random = 0;
  uint64_t p = 0;
  // unsigned p = nqubits;
  for (p = nqubits; p < 2 * nqubits; p++)
    if (table[p].X[qubit]) {
      is_random = 1;
      break;
    }
  bool outcome;
  if (is_random) {
    for (uint64_t i = 0; i < 2 * nqubits; i++)
      // the last condition is not in the AG paper but we seem to need it
      if ((table[i].X[qubit]) && (i != p) && (i != (p - nqubits)))
        rowsum(i, p);
    table[p - nqubits].X = table[p].X;
    table[p - nqubits].Z = table[p].Z;
    table[p - nqubits].phase = table[p].phase;
    table[p].X.makeZero();
    table[p].Z.makeZero();
    table[p].Z.setValue(1, qubit);
    outcome = (randint == 1);
    table[p].phase = outcome;
  } else {
    // make the auxiliary row equal to zero
    table[2 * nqubits].X.makeZero();
    table[2 * nqubits].Z.makeZero();
    table[2 * nqubits].phase = 0;
    for (uint64_t i = 0; i < nqubits; i++)
      if (table[i].X[qubit])
        rowsum(2 * nqubits, i + nqubits);
    outcome = table[2 * nqubits].phase;
  }
  return outcome;
}

bool Clifford::MeasX(const uint64_t qubit, const uint64_t randint) {
  H(qubit);
  bool b = MeasZ(qubit, randint);
  H(qubit);
  return b;
}

bool Clifford::MeasY(const uint64_t qubit, const uint64_t randint) {
  S(qubit);
  Z(qubit);
  bool b = MeasX(qubit, randint);
  S(qubit);
  return b;
}

void Clifford::PrepZ(const uint64_t qubit, const uint64_t randint) {
  bool b = MeasZ(qubit, randint);
  if (b)
    X(qubit);
}

void Clifford::PrepX(const uint64_t qubit, const uint64_t randint) {
  H(qubit);
  bool b = MeasZ(qubit, randint);
  H(qubit);
  if (b)
    Z(qubit);
}

void Clifford::PrepY(const uint64_t qubit, const uint64_t randint) {
  PrepX(qubit, randint);
  S(qubit);
}

void Clifford::Reset() {
  // prepare all-zeros state
  for (uint64_t q = 0; q < nqubits; q++) {
    // update destabilizers
    table[q].X.makeZero();
    table[q].Z.makeZero();
    table[q].phase = 0;
    table[q].X.setValue(1, q);
    // update stabilizers
    table[q + nqubits].X.makeZero();
    table[q + nqubits].Z.makeZero();
    table[q + nqubits].phase = 0;
    table[q + nqubits].Z.setValue(1, q);
  }

  // update the auxiliary row
  table[2 * nqubits].X.makeZero();
  table[2 * nqubits].Z.makeZero();
  table[2 * nqubits].phase = 0;
}

// End clifford.hpp
#endif
