/*
Copyright (c) 2018 IBM Corporation. All Rights Reserved.

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
 * @file    tensor_index.hpp
 * @brief   TensorIndex class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _qubit_index_hpp_
#define _qubit_index_hpp_

//#define DEBUG // error checking


#include <array>
#include <vector>
#include <cstdint>

// TODO -- add support for qudit indexing

namespace TI {
  
  using uint_t = uint64_t;

/*******************************************************************************
 *
 * TensorIndex Class
 *
 ******************************************************************************/

class TensorIndex {

public:

  /************************
   * Constructors
   ************************/

  TensorIndex();

  /**
   * Note that the following function requires the qubit indexes to be sorted
   * in ascending qubit order
   * Eg: qubits_sorted = {0, 1, 2}
   */
  /***************************
   * Dynamic Indexing (slower)
   ***************************/
  uint_t index0_dynamic(const std::vector<uint_t> &qubits_sorted, const size_t N,
                        const uint_t k) const;
  std::vector<uint_t> indexes_dynamic(const std::vector<uint_t> &qubitss,
                                      const std::vector<uint_t> &qubits_sorted,
                                      const size_t N, const uint_t k) const;

  /***************************
   * Static Indexing (faster)
   ***************************/
  template <size_t N>
  uint_t index0_static(const std::array<uint_t, N> &qubits_sorted, const uint_t k) const;

  template <size_t N>
  std::array<uint_t, 1ULL << N> indexes_static(const std::array<uint_t, N> &qubitss,
                                               const std::array<uint_t, N> &qubits_sorted,
                                               const uint_t k) const;
  std::array<uint_t, 4> indexes_static(const std::array<uint_t, 2> &qubits,
                                       const std::array<uint_t, 2> &qubits_sorted,
                                       const uint_t k) const;

  std::array<uint_t, 2> indexes_static(const std::array<uint_t, 1> &qubits,
                                const std::array<uint_t, 1> &qubits_sorted,
                                const uint_t k) const;

protected:
  std::array<uint_t, 64> masks;
  std::array<uint_t, 64> bits;
};


/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/


//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------

TensorIndex::TensorIndex() {
  // initialize masks
  for (uint_t i = 0; i < 64; i++)
    masks[i] = (1ULL << i) - 1;
  for (uint_t i = 0; i < 64; i++)
    bits[i] = (1ULL << i);
}

//------------------------------------------------------------------------------
// Static Indexing
//------------------------------------------------------------------------------

template <size_t N>
uint_t TensorIndex::index0_static(const std::array<uint_t, N> &qubits_sorted,
                                 const uint_t k) const {
  uint_t lowbits = 0, mask = 0;
  for (size_t j = 0; j < N; j++) {
    mask ^= masks[qubits_sorted[j] - j];
    lowbits |= (k & mask) << j;
  }
  uint_t retval = k >> (qubits_sorted[N - 1] - N + 1);
  retval <<= (qubits_sorted[N - 1] + 1);
  retval |= lowbits;
  return retval;
}

template <size_t N>
std::array<uint_t, 1ULL << N>
TensorIndex::indexes_static(const std::array<uint_t, N> &qs,
                            const std::array<uint_t, N> &qubits_sorted,
                            const uint_t k) const {
  std::array<uint_t, 1ULL << N> ret;
  ret[0] = index0_static<N>(qubits_sorted, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = 1ULL << i;
    const auto bit = bits[qs[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

std::array<uint_t, 2>
TensorIndex::indexes_static(const std::array<uint_t, 1> &qs,
                           const std::array<uint_t, 1> &qubits_sorted,
                           const uint_t k) const {
  std::array<uint_t, 2> ret;
  ret[0] = index0_static(qubits_sorted, k);
  ret[1] = ret[0] | bits[qs[0]];
  return ret;
}

std::array<uint_t, 4>
TensorIndex::indexes_static(const std::array<uint_t, 2> &qs,
                           const std::array<uint_t, 2> &qubits_sorted,
                           const uint_t k) const {
  std::array<uint_t, 4> ret;
  ret[0] = index0_static(qubits_sorted, k);
  ret[1] = ret[0] | bits[qs[0]];
  ret[2] = ret[0] | bits[qs[1]];
  ret[3] = ret[1] | bits[qs[1]];
  return ret;
}

//------------------------------------------------------------------------------
// Dynamic Indexing
//------------------------------------------------------------------------------

uint_t TensorIndex::index0_dynamic(const std::vector<uint_t> &qubits_sorted, const size_t N,
                                   const uint_t k) const {
  uint_t lowbits = 0, mask = 0;
  for (size_t j = 0; j < N; j++) {
    mask ^= masks[qubits_sorted[j] - j];
    lowbits |= (k & mask) << j;
  }
  uint_t retval = k >> (qubits_sorted[N - 1] - N + 1);
  retval <<= (qubits_sorted[N - 1] + 1);
  retval |= lowbits;
  return retval;
}

std::vector<uint_t>
TensorIndex::indexes_dynamic(const std::vector<uint_t> &qs,
                             const std::vector<uint_t> &qubits_sorted, const size_t N,
                           const uint_t k) const {
  std::vector<uint_t> ret(1ULL << N);
  ret[0] = index0_dynamic(qubits_sorted, N, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = 1ULL << i;
    const auto bit = bits[qs[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

}
//------------------------------------------------------------------------------
#endif // end module