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
 * @file    qubit_vector.hpp
 * @brief   QubitVector class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _qubit_vector_hpp_
#define _qubit_vector_hpp_

//#define DEBUG // error checking

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "tensor_index.hpp" // multipartite qubit indexing

namespace QV {

// Types
using TI::TensorIndex;
using TI::uint_t;
using omp_int_t = int64_t; // signed int for OpenMP 2.0 on msvc
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;

/*******************************************************************************
 *
 * QubitVector Class
 *
 ******************************************************************************/

class QubitVector {

public:

  /************************
   * Constructors
   ************************/

  explicit QubitVector(size_t num_qubits = 0);
  QubitVector(const cvector_t &vec);
  QubitVector(const rvector_t &vec);

  /************************
   * Utility
   ************************/

  inline uint_t size() const { return num_states;};
  inline uint_t qubits() const { return num_qubits;};
  inline cvector_t &vector() { return state_vector;};
  inline cvector_t vector() const { return state_vector;};

  complex_t dot(const QubitVector &qv) const;
  complex_t inner_product(const QubitVector &qv) const;
  double norm() const;
  void conj();
  void renormalize();
  void initialize();
  void initialize_plus();

  void set_omp_threads(int n);
  void set_omp_threshold(int n);

  /**************************************
   * Z-measurement outcome probabilities
   **************************************/

  rvector_t probabilities() const;
  rvector_t probabilities(const uint_t qubit) const;
  rvector_t probabilities(const std::vector<uint_t> &qubits) const;
  template <size_t N>
  rvector_t probabilities(const std::array<uint_t, N> &qubits) const;

  /**************************************
   * Z-measurement outcome probability
   **************************************/
  double probability(const uint_t outcome) const;
  double probability(const uint_t qubit, const uint_t outcome) const;
  double probability(const std::vector<uint_t> &qubits, const uint_t outcome) const;
  template <size_t N>
  double probability(const std::array<uint_t, N> &qubits, const uint_t outcome) const;

  /************************
   * Apply Matrices
   ************************/

  // Matrices vectorized in column-major
  void apply_matrix(const uint_t qubit, const cvector_t &mat);
  void apply_matrix(const uint_t qubit0, const uint_t qubit1, const cvector_t &mat);
  void apply_matrix(const std::vector<uint_t> &qubits, const cvector_t &mat);
  template <size_t N>
  void apply_matrix(const std::array<uint_t, N> &qubits, const cvector_t &mat);
  
  // Specialized gates
  void apply_cnot(const uint_t qctrl, const uint_t qtrgt);
  void apply_cz(const uint_t q0, const uint_t q1);
  void apply_x(const uint_t qubit);
  void apply_y(const uint_t qubit);
  void apply_z(const uint_t qubit);

  /************************
   * Norms
   ************************/

  double norm(const uint_t qubit, const cvector_t &mat) const;
  double norm(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  template <size_t N>
  double norm(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;
  
  /************************
   * Expectation Values
   ************************/

  complex_t expectation_value(const uint_t qubit, const cvector_t &mat) const;
  complex_t expectation_value(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  template <size_t N>
  complex_t expectation_value(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;


  /************************
   * Operators
   ************************/

  // Assignment operator
  QubitVector &operator=(const cvector_t &vec);
  QubitVector &operator=(const rvector_t &vec);

  // Element access
  complex_t &operator[](uint_t element);
  complex_t operator[](uint_t element) const;
  
  // Scalar multiplication
  QubitVector &operator*=(const complex_t &lambda);
  QubitVector &operator*=(const double &lambda);
  friend QubitVector operator*(const complex_t &lambda, const QubitVector &qv);
  friend QubitVector operator*(const double &lambda, const QubitVector &qv);
  friend QubitVector operator*(const QubitVector &qv, const complex_t &lambda);
  friend QubitVector operator*(const QubitVector &qv, const double &lambda);
  
  // Vector addition
  QubitVector &operator+=(const QubitVector &qv);
  QubitVector operator+(const QubitVector &qv) const;
  
  // Vector subtraction
  QubitVector &operator-=(const QubitVector &qv);
  QubitVector operator-(const QubitVector &qv) const;

protected:
  size_t num_qubits;
  size_t num_states;
  cvector_t state_vector;
  TensorIndex idx;

  // OMP
  uint_t omp_threads = 1;     // Disable multithreading by default
  uint_t omp_threshold = 16;  // Qubit threshold for multithreading when enabled

  /************************
   * Matrix-mult Helper functions
   ************************/

  void apply_matrix_col_major(const uint_t qubit, const cvector_t &mat);
  void apply_matrix_col_major(const std::vector<uint_t> &qubits, const cvector_t &mat);
  template <size_t N>
  void apply_matrix_col_major(const std::array<uint_t, N> &qubits, const cvector_t &mat);

  void apply_matrix_diagonal(const uint_t qubit, const cvector_t &mat);
  void apply_matrix_diagonal(const std::vector<uint_t> &qubits, const cvector_t &mat);
  template <size_t N>
  void apply_matrix_diagonal(const std::array<uint_t, N> &qubits, const cvector_t &mat);

  // Norms
  // Warning: no test coverage
  double norm_matrix(const uint_t qubit, const cvector_t &mat) const;
  double norm_matrix_diagonal(const uint_t qubit, const cvector_t &mat) const;
  double norm_matrix(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  double norm_matrix_diagonal(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  template <size_t N>
  double norm_matrix(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;
  template <size_t N>
  double norm_matrix_diagonal(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;

  // Matrix Expectation Values
  // Warning: no test coverage
  complex_t expectation_value_matrix(const uint_t qubit, const cvector_t &mat) const;
  complex_t expectation_value_matrix_diagonal(const uint_t qubit, const cvector_t &mat) const;
  complex_t expectation_value_matrix(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  complex_t expectation_value_matrix_diagonal(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  template <size_t N>
  complex_t expectation_value_matrix(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;
  template <size_t N>
  complex_t expectation_value_matrix_diagonal(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;

  // Error messages
  void check_qubit(const uint_t qubit) const;
  void check_vector(const cvector_t &diag, uint_t nqubits) const;
  void check_matrix(const cvector_t &mat, uint_t nqubits) const;
  void check_dimension(const QubitVector &qv) const;

};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

void QubitVector::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits) {
    std::stringstream ss;
    ss << "QubitVector: qubit index " << qubit << " > " << num_qubits;
    throw std::runtime_error(ss.str());
  }
}

void QubitVector::check_matrix(const cvector_t &vec, uint_t nqubits) const {
  const size_t dim = 1ULL << nqubits;
  const auto sz = vec.size();
  if (sz != dim * dim) {
    std::stringstream ss;
    ss << "QubitVector: vector size is " << sz << " != " << (dim * dim);
    throw std::runtime_error(ss.str());
  }
}

void QubitVector::check_vector(const cvector_t &vec, uint_t nqubits) const {
  const size_t dim = 1ULL << nqubits;
  const auto sz = vec.size();
  if (sz != dim) {
    std::stringstream ss;
    ss << "QubitVector: vector size is " << sz << " != " << dim;
    throw std::runtime_error(ss.str());
  }
}

void QubitVector::check_dimension(const QubitVector &qv) const {
  if (num_states != qv.num_states) {
    std::stringstream ss;
    ss << "QubitVector: vectors are different size ";
    ss << num_states << " != " << qv.num_states;
    throw std::runtime_error(ss.str());
  }
}

//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------

QubitVector::QubitVector(size_t num_qubits_) : num_qubits(num_qubits_),
                                               num_states(1ULL << num_qubits_),
                                               idx()  {
  // Set state vector
  state_vector.assign(num_states, 0.);
}

QubitVector::QubitVector(const cvector_t &vec) : QubitVector() {
  *this = vec;
}

QubitVector::QubitVector(const rvector_t &vec) : QubitVector() {
  *this = vec;
}


//------------------------------------------------------------------------------
// Operators
//------------------------------------------------------------------------------

// Access opertors

complex_t &QubitVector::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  auto size = state_vector.size();
  if (element > size) {
    std::stringstream ss;
    ss << "QubitVector: vector index " << element << " > " << size;
    throw std::runtime_error(ss.str());
  }
  #endif
  return state_vector[element];
}
  

complex_t QubitVector::operator[](uint_t element) const {
  // Error checking
  #ifdef DEBUG
  auto size = state_vector.size();
  if (element > size) {
    std::stringstream ss;
    ss << "QubitVector: vector index " << element << " > " << size;
    throw std::runtime_error(ss.str());
  }
  #endif
  return state_vector[element];
}
  
// Equal operators
QubitVector &QubitVector::operator=(const cvector_t &vec) {
  
  num_states = vec.size();
  // Get qubit number
  uint_t size = num_states;
  num_qubits = 0;
  while (size >>= 1) ++num_qubits;
  
  // Error handling
  #ifdef DEBUG
    if (num_states != 1ULL << num_qubits) {
      std::stringstream ss;
      ss << "QubitVector: input vector is not a multi-qubit vector.";
      throw std::runtime_error(ss.str());
    }
  #endif
  // Set state_vector
  state_vector = vec;
  return *this;
}

QubitVector &QubitVector::operator=(const rvector_t &vec) {
  
  num_states = vec.size();
  // Get qubit number
  uint_t size = num_states;
  num_qubits = 0;
  while (size >>= 1) ++num_qubits;
  
  // Error handling
  #ifdef DEBUG
    if (num_states != 1ULL << num_qubits) {
      std::stringstream ss;
      ss << "QubitVector: input vector is not a multi-qubit vector.";
      throw std::runtime_error(ss.str());
    }
  #endif
  // Set state_vector
  state_vector.clear();
  state_vector.reserve(size);
  for (const auto& v: vec)
    state_vector.push_back(v);
  return *this;
}

// Scalar multiplication
QubitVector &QubitVector::operator*=(const complex_t &lambda) {
const omp_int_t end = num_states;    // end for k loop
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++)
      state_vector[k] *= lambda;
  } // end omp parallel
  return *this;
}

QubitVector &QubitVector::operator*=(const double &lambda) {
  *this *= complex_t(lambda);
  return *this;
}

QubitVector operator*(const complex_t &lambda, const QubitVector &qv) {
  QubitVector ret = qv;
  ret *= lambda;
  return ret;
}

QubitVector operator*(const QubitVector &qv, const complex_t &lambda) {
  return lambda * qv;
}

QubitVector operator*(const double &lambda, const QubitVector &qv) {
  return complex_t(lambda) * qv;
}

QubitVector operator*(const QubitVector &qv, const double &lambda) {
  return lambda * qv;
}

// Vector addition

QubitVector &QubitVector::operator+=(const QubitVector &qv) {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif
  const omp_int_t end = num_states;    // end for k loop
  #pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++)
      state_vector[k] += qv.state_vector[k];
  } // end omp parallel
  return *this;
}

QubitVector QubitVector::operator+(const QubitVector &qv) const{
  QubitVector ret = *this;
  ret += qv;
  return ret;
}

// Vector subtraction

QubitVector &QubitVector::operator-=(const QubitVector &qv) {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif
  const omp_int_t end = num_states;    // end for k loop
  #pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++)
      state_vector[k] -= qv.state_vector[k];
  } // end omp parallel
  return *this;
}

QubitVector QubitVector::operator-(const QubitVector &qv) const{
  QubitVector ret = *this;
  ret -= qv;
  return ret;
}


//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

void QubitVector::initialize() {
  state_vector.assign(num_states, 0.);
  state_vector[0] = 1.;
}

void QubitVector::initialize_plus() {
  complex_t val(1.0 / std::pow(2, 0.5 * num_qubits), 0.);
  state_vector.assign(num_states, val);
}

void QubitVector::conj() {
  const omp_int_t end = num_states;    // end for k loop
  #pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      state_vector[k] = std::conj(state_vector[k]);
    }
  } // end omp parallel
}

complex_t QubitVector::dot(const QubitVector &qv) const {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif

// split variable for OpenMP 2.0 compatible reduction
double z_re = 0., z_im = 0.;
const omp_int_t end = num_states;    // end for k loop
#pragma omp parallel reduction(+:z_re, z_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      const complex_t z = state_vector[k] * qv.state_vector[k];
      z_re += std::real(z);
      z_im += std::imag(z);
    }
  } // end omp parallel
  return complex_t(z_re, z_im);
}

complex_t QubitVector::inner_product(const QubitVector &qv) const {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif

double z_re = 0., z_im = 0.;
const omp_int_t end = num_states;    // end for k loop
#pragma omp parallel reduction(+:z_re, z_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      const complex_t z = state_vector[k] * std::conj(qv.state_vector[k]);
      z_re += std::real(z);
      z_im += std::imag(z);
    }
  } // end omp parallel
  return complex_t(z_re, z_im);
}

void QubitVector::renormalize() {
  double nrm = norm();
  #ifdef DEBUG
    if ((nrm > 0.) == false) {
      std::stringstream ss;
      ss << "QubitVector: vector has norm zero.";
      throw std::runtime_error(ss.str());
    }
  #endif
  if (nrm > 0.) {
    const double scale = 1.0 / std::sqrt(nrm);
    *this *= scale;
  }
}

void QubitVector::set_omp_threads(int n) {
  if (n > 0)
    omp_threads = n;
}

void QubitVector::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold = n;
}


/*******************************************************************************
 *
 * SINGLE QUBIT OPERATIONS
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Matrix multiplication
//------------------------------------------------------------------------------

void QubitVector::apply_matrix(const uint_t qubit, const cvector_t &mat) {
  if (mat.size() == 2)
    apply_matrix_diagonal(qubit, mat);
  else
    apply_matrix_col_major(qubit, mat);
}

void QubitVector::apply_matrix_col_major(const uint_t qubit, const cvector_t &mat) {
  
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2);
  #endif

  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = state_vector[k];
        const auto cache1 = state_vector[k | end2];
        state_vector[k] = mat[0] * cache0 + mat[2] * cache1;
        state_vector[k | end2] = mat[1] * cache0 + mat[3] * cache1;
      }
  }
}

void QubitVector::apply_matrix_diagonal(const uint_t qubit, const cvector_t &diag) {
  
  // Error checking
  #ifdef DEBUG
  check_vector(diag, 1);
  check_qubit(qubit);
  #endif
  
  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        state_vector[k] *= diag[0];
        state_vector[k | end2] *= diag[1];
      }
  }
}

void QubitVector::apply_x(const uint_t qubit) {
  
  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  // Optimized ideal Pauli-X gate
  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto i0 = k1 | k2;
        const auto i1 = i0 | end2;
        const complex_t cache = state_vector[i0];
        state_vector[i0] = state_vector[i1]; // mat(0,1)
        state_vector[i1] = cache;    // mat(1,0)
      }
  }
}

void QubitVector::apply_y(const uint_t qubit) {
 // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  // Optimized ideal Pauli-Y gate
  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
  const complex_t I(0., 1.);
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto i0 = k1 | k2;
        const auto i1 = i0 | end2;
        const complex_t cache = state_vector[i0];
        state_vector[i0] = -I * state_vector[i1]; // mat(0,1)
        state_vector[i1] = I * cache;     // mat(1,0)
      }
  }
}

void QubitVector::apply_z(const uint_t qubit) {
  
  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif
  
  // Optimized ideal Pauli-Z gate
  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
  const complex_t minus_one(-1.0, 0.0);
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        state_vector[k1 | k2 | end2] *= minus_one;
      }
  }
}


//------------------------------------------------------------------------------
// Norm
//------------------------------------------------------------------------------


double QubitVector::norm() const {
  double val = 0;
  const omp_int_t end = num_states;    // end for k loop
  #pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++)
      val += std::real(state_vector[k] * std::conj(state_vector[k]));
  } // end omp parallel
  return val;
}

double QubitVector::norm(const uint_t qubit, const cvector_t &mat) const {
  if (mat.size() == 2)
      return norm_matrix_diagonal(qubit, mat);
  else
      return norm_matrix(qubit, mat);
}

double QubitVector::norm_matrix(const uint_t qubit, const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  check_vector(mat, 2);
  #endif

  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
  double val = 0.;
#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = state_vector[k];
        const auto cache1 = state_vector[k | end2];
        const auto v0 = mat[0] * cache0 + mat[2] * cache1;
        const auto v1 = mat[1] * cache0 + mat[3] * cache1;
        val += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
      }
  } // end omp parallel
  return val;
}

double QubitVector::norm_matrix_diagonal(const uint_t qubit, const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  check_vector(mat, 1);
  #endif

  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
  double val = 0.;
#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto v0 = mat[0] * state_vector[k];
        const auto v1 = mat[1] * state_vector[k | end2];
        val += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
      }
  } // end omp parallel
  return val;
}


//------------------------------------------------------------------------------
// Expectation Values
//------------------------------------------------------------------------------

complex_t QubitVector::expectation_value(const uint_t qubit, const cvector_t &mat) const {
  if (mat.size() == 2)
    return expectation_value_matrix_diagonal(qubit, mat);
  else
    return expectation_value_matrix(qubit, mat);
}

complex_t QubitVector::expectation_value_matrix(const uint_t qubit, const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  check_vector(mat, 2);
  #endif

  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = state_vector[k];
        const auto cache1 = state_vector[k | end2];
        const auto v0 = mat[0] * cache0 + mat[2] * cache1;
        const auto v1 = mat[1] * cache0 + mat[3] * cache1;
        const complex_t val = v0 * std::conj(cache0) + v1 * std::conj(cache1);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

complex_t QubitVector::expectation_value_matrix_diagonal(const uint_t qubit, const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  check_vector(mat, 1);
  #endif

  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = state_vector[k];
        const auto cache1 = state_vector[k | end2];
        const complex_t val = mat[0] * cache0 * std::conj(cache0) + mat[1] * cache1 * std::conj(cache1);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


/*******************************************************************************
 *
 * STATIC N-QUBIT OPERATIONS (N known at compile time)
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Matrix multiplication
//------------------------------------------------------------------------------

void QubitVector::apply_matrix(const uint_t qubit0, const uint_t qubit1,
                               const cvector_t &mat) {
  if (mat.size() == 4)
    apply_matrix_diagonal<2>({{qubit0, qubit1}}, mat);
  else
    apply_matrix_col_major<2>({{qubit0, qubit1}}, mat);
}

template <size_t N>
void QubitVector::apply_matrix(const std::array<uint_t, N> &qs, const cvector_t &mat) {
  if (mat.size() == (1ULL << N))
    apply_matrix_diagonal<N>(qs, mat);
  else
    apply_matrix_col_major<N>(qs, mat);
}

template <size_t N>
void QubitVector::apply_matrix_col_major(const std::array<uint_t, N> &qs,
                               const cvector_t &mat) {
  
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_static(qs, qubits_sorted, k);
      std::array<complex_t, dim> cache;
      for (size_t i = 0; i < dim; i++) {
        const auto ii = inds[i];
        cache[i] = state_vector[ii];
        state_vector[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
          state_vector[inds[i]] += mat[i + dim * j] * cache[j];
    }
  }
}

template <size_t N>
void QubitVector::apply_matrix_diagonal(const std::array<uint_t, N> &qs,
                               const cvector_t &diag) {
  
  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      const auto inds = idx.indexes_static(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++)
          state_vector[inds[i]] *= diag[i];
    }
  }
}

void QubitVector::apply_cnot(const uint_t qubit_ctrl, const uint_t qubit_trgt) {
  
  // Error checking
  #ifdef DEBUG
  check_qubit(qubit_ctrl);
  check_qubit(qubit_trgt);
  #endif

  const omp_int_t end = num_states >> 2;
  const auto qubits_sorted = (qubit_ctrl < qubit_trgt)
                          ? std::array<uint_t, 2>{{qubit_ctrl, qubit_trgt}}
                          : std::array<uint_t, 2>{{qubit_trgt, qubit_ctrl}};

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {

      const auto ii = idx.indexes_static<2>({{qubit_ctrl, qubit_trgt}},
                                            qubits_sorted, k);
      const complex_t cache = state_vector[ii[3]];
      state_vector[ii[3]] = state_vector[ii[1]];
      state_vector[ii[1]] = cache;
    }
  } // end omp parallel
}

void QubitVector::apply_cz(const uint_t qubit_ctrl, const uint_t qubit_trgt) {

  // Error checking
  #ifdef DEBUG
  check_qubit(qubit_ctrl);
  check_qubit(qubit_trgt);
  #endif

  const omp_int_t end = num_states >> 2;
  const auto qubits_sorted = (qubit_ctrl < qubit_trgt)
                          ? std::array<uint_t, 2>{{qubit_ctrl, qubit_trgt}}
                          : std::array<uint_t, 2>{{qubit_trgt, qubit_ctrl}};

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      const auto ii = idx.indexes_static<2>({{qubit_ctrl, qubit_trgt}},
                                            qubits_sorted, k);
      state_vector[ii[3]] *= -1.;
    }
  }
}


//------------------------------------------------------------------------------
// Norm
//------------------------------------------------------------------------------

template <size_t N>
double QubitVector::norm(const std::array<uint_t, N> &qs, const cvector_t &mat) const {
  if (mat.size() == (1ULL << N))
    return norm_matrix_diagonal<N>(qs, mat);
  else
    return norm_matrix<N>(qs, mat);
}

template <size_t N>
double QubitVector::norm_matrix(const std::array<uint_t, N> &qs, const cvector_t &mat) const {
  
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val = 0.;

#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_static(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++)
          vi += mat[i + dim * j] * state_vector[inds[j]];
        val += std::real(vi * std::conj(vi));
      }
    }
  } // end omp parallel
  return val;
}


template <size_t N>
double QubitVector::norm_matrix_diagonal(const std::array<uint_t, N> &qs, const cvector_t &mat) const {
  
  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val = 0.;
#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_static(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++) {
        const auto vi = mat[i] * state_vector[inds[i]];
        val += std::real(vi * std::conj(vi));
      }
    }
  } // end omp parallel
  return val;
}

//------------------------------------------------------------------------------
// Expectation Values
//------------------------------------------------------------------------------

template <size_t N>
complex_t QubitVector::expectation_value(const std::array<uint_t, N> &qs, const cvector_t &mat) const {
  if (mat.size() == (1ULL << N))
    return expectation_value_matrix_diagonal<N>(qs, mat);
  else
    return expectation_value_matrix<N>(qs, mat);
}

template <size_t N>
complex_t QubitVector::expectation_value_matrix(const std::array<uint_t, N> &qs, const cvector_t &mat) const {
  
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_static(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++) {
          vi += mat[i + dim * j] * state_vector[inds[j]];
        }
        const complex_t val = vi * std::conj(state_vector[inds[i]]);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


template <size_t N>
complex_t QubitVector::expectation_value_matrix_diagonal(const std::array<uint_t, N> &qs, const cvector_t &mat) const {
  
  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_static(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++) {
        const auto cache = state_vector[inds[i]];
        const complex_t val = mat[i] * cache * std::conj(cache);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


/*******************************************************************************
 *
 * DYNAMIC N-QUBIT OPERATIONS (N known at run time)
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Matrix multiplication
//------------------------------------------------------------------------------

void QubitVector::apply_matrix(const std::vector<uint_t> &qs, const cvector_t &mat) {
  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    apply_matrix<1>(std::array<uint_t, 1>({{qs[0]}}), mat);
    break;
  case 2:
    apply_matrix<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), mat);
    break;
  case 3:
    apply_matrix<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), mat);
    break;
  case 4:
    apply_matrix<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), mat);
    break;
  case 5:
    apply_matrix<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), mat);
    break;
  }
  // General case
  if (mat.size() == (1ULL << qs.size()))
    apply_matrix_diagonal(qs, mat);
  else
    apply_matrix_col_major(qs, mat);
}

void QubitVector::apply_matrix_col_major(const std::vector<uint_t> &qubits, const cvector_t &mat) {

  const auto N = qubits.size();
  const uint_t dim = 1ULL << N;
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_dynamic(qubits, qubits_sorted, N, k);
      std::vector<complex_t> cache(dim);
      for (size_t i = 0; i < dim; i++) {
        const auto ii = inds[i];
        cache[i] = state_vector[ii];
        state_vector[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
          state_vector[inds[i]] += mat[i + dim * j] * cache[j];
    }
  }
}

void QubitVector::apply_matrix_diagonal(const std::vector<uint_t> &qubits,
                               const cvector_t &diag) {
  
  const auto N = qubits.size();
  const uint_t dim = 1ULL << N;
  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      const auto inds = idx.indexes_dynamic(qubits, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++)
          state_vector[inds[i]] *= diag[i];
    }
  }
}


//------------------------------------------------------------------------------
// Norm
//------------------------------------------------------------------------------

double QubitVector::norm(const std::vector<uint_t> &qs, const cvector_t &mat) const {
  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    return norm<1>(std::array<uint_t, 1>({{qs[0]}}), mat);
  case 2:
    return norm<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), mat);
  case 3:
    return norm<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), mat);
  case 4:
    return norm<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), mat);
  case 5:
    return norm<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), mat);
  }
  // General case
  if (mat.size() == (1ULL << qs.size()))
    return norm_matrix_diagonal(qs, mat);
  else
    return norm_matrix(qs, mat);
}

double QubitVector::norm_matrix(const std::vector<uint_t> &qs, const cvector_t &mat) const {
  
  // Error checking
  const uint_t N = qs.size();
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val = 0.;

#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++)
          vi += mat[i + dim * j] * state_vector[inds[j]];
        val += std::real(vi * std::conj(vi));
      }
    }
  } // end omp parallel
  return val;
}

double QubitVector::norm_matrix_diagonal(const std::vector<uint_t> &qs, const cvector_t &mat) const {
  
  // Error checking
  const uint_t N = qs.size();
  #ifdef DEBUG
  check_vector(mat, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val = 0.;
#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++) {
        const auto vi = mat[i] * state_vector[inds[i]];
        val += std::real(vi * std::conj(vi));
      }
    }
  } // end omp parallel
  return val;
}


//------------------------------------------------------------------------------
// Expectation Values
//------------------------------------------------------------------------------

complex_t QubitVector::expectation_value(const std::vector<uint_t> &qs, const cvector_t &mat) const {
  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    return expectation_value<1>(std::array<uint_t, 1>({{qs[0]}}), mat);
  case 2:
    return expectation_value<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), mat);
  case 3:
    return expectation_value<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), mat);
  case 4:
    return expectation_value<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), mat);
  case 5:
    return expectation_value<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), mat);
  }
  // General case
  if (mat.size() == (1ULL << qs.size()))
    return expectation_value_matrix_diagonal(qs, mat);
  else
    return expectation_value_matrix(qs, mat);
}

complex_t QubitVector::expectation_value_matrix(const std::vector<uint_t> &qs, const cvector_t &mat) const {
  
  // Error checking
  const uint_t N = qs.size();
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++) {
          vi += mat[i + dim * j] * state_vector[inds[j]];
        }
        const complex_t val = vi * std::conj(state_vector[inds[i]]);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

complex_t QubitVector::expectation_value_matrix_diagonal(const std::vector<uint_t> &qs, const cvector_t &mat) const {
  
  // Error checking
  const uint_t N = qs.size();
  #ifdef DEBUG
  check_vector(mat, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++) {
        const auto cache = state_vector[inds[i]];
        const complex_t val = mat[i] * cache * std::conj(cache);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// All outcome probabilities
//------------------------------------------------------------------------------

rvector_t QubitVector::probabilities() const {
  rvector_t probs;
  probs.reserve(num_states);
  const omp_int_t end = state_vector.size();
  for (omp_int_t j=0; j < end; j++) {
    probs.push_back(probability(j));
  }
  return probs;
}

rvector_t QubitVector::probabilities(const uint_t qubit) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
  double p0 = 0., p1 = 0.;
#pragma omp parallel reduction(+:p0, p1) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        p0 += probability(k);
        p1 += probability(k | end2);
      } 
  } // end omp parallel
  return rvector_t({p0, p1});
}

template <size_t N>
rvector_t QubitVector::probabilities(const std::array<uint_t, N> &qs) const {
  
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  if (N == 0)
    return rvector_t({norm()});
  
  const uint_t dim = 1ULL << N;
  const uint_t end = (1ULL << num_qubits) >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  if ((N == num_qubits) && (qs == qss))
    return probabilities();

  rvector_t probs(dim, 0.);
  for (size_t k = 0; k < end; k++) {
    const auto indexes = idx.indexes_static<N>(qs, qubits_sorted, k);
    for (size_t m = 0; m < dim; ++m) {
      probs[m] += probability(indexes[m]);
    }
  }
  return probs;
}

rvector_t QubitVector::probabilities(const std::vector<uint_t> &qs) const {

  // Special cases using faster static indexing
  const uint_t N = qs.size();
  switch (N) {
  case 0:
    return rvector_t({norm()});
  case 1:
    return probabilities<1>(std::array<uint_t, 1>({{qs[0]}}));
  case 2:
    return probabilities<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}));
  case 3:
    return probabilities<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}));
  case 4:
    return probabilities<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}));
  case 5:
    return probabilities<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}));
  }
  // else
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const uint_t dim = 1ULL << N;
  const uint_t end = (1ULL << num_qubits) >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  if ((N == num_qubits) && (qss == qs))
    return probabilities();
  const auto &qubits_sorted = qss;
  rvector_t probs(dim, 0.);

  for (size_t k = 0; k < end; k++) {
    const auto indexes = idx.indexes_dynamic(qs, qubits_sorted, N, k);
    for (size_t m = 0; m < dim; ++m)
      probs[m] += probability(indexes[m]);
  }
  return probs;
}

//------------------------------------------------------------------------------
// Single outcome probability
//------------------------------------------------------------------------------
double QubitVector::probability(const uint_t outcome) const {
  const auto v = state_vector[outcome];
  return std::real(v * std::conj(v));
}

double QubitVector::probability(const uint_t qubit, const uint_t outcome) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const omp_int_t end1 = num_states;    // end for k1 loop
  const omp_int_t end2 = 1LL << qubit; // end for k2 loop
  const omp_int_t step1 = end2 << 1;    // step for k1 loop
  double p = 0.;
#pragma omp parallel reduction(+:p) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
  if (outcome == 0) {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++)
        p += probability(k1 | k2);
  } else if (outcome == 1) {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (omp_int_t k1 = 0; k1 < end1; k1 += step1)
      for (omp_int_t k2 = 0; k2 < end2; k2++)
        p += probability(k1 | k2 | end2);
  }
  } // end omp parallel
  return p;
}

template <size_t N>
double QubitVector::probability(const std::array<uint_t, N> &qs,
                                const uint_t outcome) const {
  
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = (1ULL << num_qubits) >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double p = 0.;

#pragma omp parallel reduction(+:p) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++)
      p += probability(idx.indexes_static<N>(qs, qubits_sorted, k)[outcome]);
  }
  return p;
}

double QubitVector::probability(const std::vector<uint_t> &qs,
                                const uint_t outcome) const {
  
  // Special cases using faster static indexing
  const uint_t N = qs.size();
  switch (N) {
  case 0:
    return norm();
  case 1:
    return probability<1>(std::array<uint_t, 1>({{qs[0]}}), outcome);
  case 2:
    return probability<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), outcome);
  case 3:
    return probability<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), outcome);
  case 4:
    return probability<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), outcome);
  case 5:
    return probability<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), outcome);
  }
  // else
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const omp_int_t end = (1ULL << num_qubits) >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double p = 0.;

#pragma omp parallel reduction(+:p) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (omp_int_t k = 0; k < end; k++)
      p += probability(idx.indexes_dynamic(qs, qubits_sorted, N, k)[outcome]);
  }
  return p;
}
//------------------------------------------------------------------------------
} // end namespace QV
#endif // end module