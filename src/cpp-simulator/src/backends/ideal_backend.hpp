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
 * @file    ideal_backend.hpp
 * @brief   Standard Qubit QISKIT simulator backend
 * @authors Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _IdealBackend_hpp_
#define _IdealBackend_hpp_

#include <algorithm>
#include <array>
#include <complex>
#include <string>
#include <vector>

#include "base_backend.hpp"

namespace QISKIT {

/*******************************************************************************
 *
 * IdealBackend class
 *
 ******************************************************************************/

class IdealBackend : public BaseBackend<cvector_t> {

public:
  /************************
   * Constructors
   ************************/

  IdealBackend() : BaseBackend<cvector_t>(){};

  /************************
   * BaseBackend Methods
   ************************/
  virtual void initialize(const Circuit &prog);
  virtual void qc_operation(const operation &op);

  /************************
   * GateSet
   ************************/

  const static gateset_t gateset;

  /************************
   * Measurement probabilities
   ************************/
  /*template <size_t N>
  std::discrete_distribution<> measure_probs(const std::array<uint_t, N> qs,
                                             const cvector_t &state);
                                             */

protected:
  MultiPartiteIndex idx; // Indexing class
  uint_t nstates;        // dimension of wavefunction

  /************************
   * Apply matrices
   ************************/
  void qc_matrix1(const uint_t qubit, const cmatrix_t &U);
  inline void qc_matrix2(const uint_t q0, const uint_t q1, const cmatrix_t &U) {
    qc_matrix<2>({{q0, q1}}, U);
  };
  template <size_t N>
  void qc_matrix(const std::array<uint_t, N> qs, const cmatrix_t &U);

  /************************
   * Measurement and Reset
   ************************/

  virtual void qc_reset(const uint_t qubit, const uint_t state = 0);
  virtual void qc_measure(const uint_t qubit, const uint_t bit);
  virtual std::pair<uint_t, double> qc_measure_outcome(const uint_t qubit);
  virtual void qc_measure_reset(const uint_t qubit, const uint_t reset_state,
                                const std::pair<uint_t, double> meas_outcome);

  /************************
   * 1-Qubit Gates
   ************************/
  virtual cmatrix_t waltz_matrix(const double theta, const double phi,
                                 const double lambda);
  virtual void qc_gate(const uint_t qubit, const double theta, const double phi,
                       const double lambda);
  virtual void qc_gate_x(const uint_t qubit);
  virtual void qc_gate_y(const uint_t qubit);
  virtual void qc_phase(const uint_t qubit, const complex_t phase);
  virtual void qc_zrot(const uint_t qubit, const double lambda);

  /************************
   * 2-Qubit Gates
   ************************/
  virtual void qc_cnot(const uint_t qctrl, const uint_t qtrgt);
  virtual void qc_cz(const uint_t q0, const uint_t q1);
  virtual void qc_zzrot(const uint_t q0, const uint_t q1, double lambda);
};

/*******************************************************************************
 *
 * Convert from JSON
 *
 ******************************************************************************/

inline void from_json(const json_t &config, IdealBackend &be) {
  be = IdealBackend();

  // Set OMP threshold for state update functions
  uint_t threshold = 20;
  JSON::get_value(threshold, "theshold_threads_gates", config);
  be.set_omp_threshold(threshold);

  // parse initial state from JSON
  if (JSON::check_key("initial_state", config)) {
    cvector_t initial_state = config["initial_state"];
    bool renorm_initial_state = true;
    JSON::get_value(renorm_initial_state, "renorm", config);
    JSON::get_value(renorm_initial_state, "renorm_initial_state", config);
    if (renorm_initial_state)
      renormalize(initial_state);
    if (initial_state.empty() == false)
      be.set_initial_state(initial_state);
  }
}

/*******************************************************************************
 *
 * BaseBackend methods
 *
 ******************************************************************************/

void IdealBackend::initialize(const Circuit &prog) {

  // system parameters
  omp_flag = (prog.nqubits > omp_threshold); // OpenMP threshold
  nstates = 1ULL << prog.nqubits;

  creg.assign(prog.nclbits, 0);
  qreg_saved.erase(qreg_saved.begin(), qreg_saved.end());

  if (qreg_init_flag) {
    if (qreg_init.size() == nstates)
      // reset state std::vector to custom state
      qreg = qreg_init;
    else {
      std::string msg = "initial state is wong size for the circuit";
      throw std::runtime_error(msg);
    }
  } else {
    // reset state std::vector to default state
    qreg.assign(nstates, 0.);
    qreg[0] = 1.;
  }
}

void IdealBackend::qc_operation(const operation &op) {

#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_operation";
  std::clog << ss.str() << std::endl;
#endif
  switch (op.id) {
  // Base gates
  case gate_t::U:
    qc_gate(op.qubits[0], op.params[0], op.params[1], op.params[2]);
    break;
  case gate_t::CX:
    qc_cnot(op.qubits[0], op.qubits[1]);
    break;
  case gate_t::Measure:
    qc_measure(op.qubits[0], op.clbits[0]);
    break;
  case gate_t::Reset:
    qc_reset(op.qubits[0], 0);
    break;
  case gate_t::Barrier:
    break;
  // Waltz gates
  case gate_t::U0: // u0 = id in ideal backend
    break;
  case gate_t::U1: // u1 = Rz(lambda) up to global phase
    qc_zrot(op.qubits[0], op.params[0]);
    break;
  case gate_t::U2: // u2 = Rz(phi)*X90*Rz(lambda)
    qc_gate(op.qubits[0], M_PI / 2., op.params[0], op.params[1]);
    break;
  case gate_t::U3: // u3 = Rz(phi)*X90*Rz(lambda)
    qc_gate(op.qubits[0], op.params[0], op.params[1], op.params[2]);
    break;
  // QIP gates
  case gate_t::I:
    break;
  case gate_t::X:
    qc_gate_x(op.qubits[0]);
    break;
  case gate_t::Y:
    qc_gate_y(op.qubits[0]);
    break;
  case gate_t::Z:
    qc_phase(op.qubits[0], -1.);
    break;
  case gate_t::H:
    qc_gate(op.qubits[0], M_PI / 2., 0., M_PI);
    break;
  case gate_t::S:
    qc_phase(op.qubits[0], complex_t(0., 1.));
    break;
  case gate_t::Sd:
    qc_phase(op.qubits[0], complex_t(0., -1.));
    break;
  case gate_t::T: {
    const double sqrt2 = 1. / std::sqrt(2.);
    qc_phase(op.qubits[0], complex_t(sqrt2, sqrt2));
  } break;
  case gate_t::Td: {
    const double sqrt2 = 1. / std::sqrt(2.);
    qc_phase(op.qubits[0], complex_t(sqrt2, -sqrt2));
  } break;
  case gate_t::CZ:
    qc_cz(op.qubits[0], op.qubits[1]);
    break;
  // ZZ rotation by angle lambda
  case gate_t::UZZ:
    qc_zzrot(op.qubits[0], op.qubits[1], op.params[0]);
    break;
  case gate_t::Wait:
    break;
  // Commands
  case gate_t::Save:
    save_state(op.params[0]);
    break;
  case gate_t::Load:
    load_state(op.params[0]);
    break;
  case gate_t::Noise:
    break;
  // Invalid Gate (we shouldn't get here)
  default:
    std::string msg = "invalid IdealBackend operation";
    throw std::runtime_error(msg);
  }
}

/*******************************************************************************
 *
 * IdealBackend members
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Static member gateset
//------------------------------------------------------------------------------

const gateset_t IdealBackend::gateset({// Core gates
                                       {"U", gate_t::U},
                                       {"CX", gate_t::CX},
                                       {"measure", gate_t::Measure},
                                       {"reset", gate_t::Reset},
                                       {"barrier", gate_t::Barrier},
                                       // Single qubit gates
                                       {"id", gate_t::I},
                                       {"x", gate_t::X},
                                       {"y", gate_t::Y},
                                       {"z", gate_t::Z},
                                       {"h", gate_t::H},
                                       {"s", gate_t::S},
                                       {"sdg", gate_t::Sd},
                                       {"t", gate_t::T},
                                       {"tdg", gate_t::Td},
                                       {"wait", gate_t::Wait},
                                       // Waltz Gates
                                       {"u0", gate_t::U0},
                                       {"u1", gate_t::U1},
                                       {"u2", gate_t::U2},
                                       {"u3", gate_t::U3},
                                       // Two-qubit gates
                                       {"cx", gate_t::CX},
                                       {"cz", gate_t::CZ},
                                       {"uzz", gate_t::UZZ},
                                       // Simulator commands
                                       {"noise", gate_t::Noise},
                                       {"save", gate_t::Save},
                                       {"load", gate_t::Load}});

//------------------------------------------------------------------------------
// Unitary Matrices
//------------------------------------------------------------------------------

void IdealBackend::qc_matrix1(const uint_t qubit, const cmatrix_t &U) {
// apply an arbitary 1-qubit operator to a qubit
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_matrix1(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end2 = 1ULL << qubit; // end for k2 loop
  const uint_t step1 = end2 << 1;    // step for k1 loop
#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for collapse(2)
    for (uint_t k1 = 0; k1 < nstates; k1 += step1)
      for (uint_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = qreg[k];
        const auto cache1 = qreg[k | end2];
        qreg[k] = U(0, 0) * cache0 + U(0, 1) * cache1;
        qreg[k | end2] = U(1, 0) * cache0 + U(1, 1) * cache1;
      }
  }
}

template <size_t N>
void IdealBackend::qc_matrix(const std::array<uint_t, N> qs,
                             const cmatrix_t &U) {

#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_matrix<" << N << ">(" << qs << ")";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end = nstates >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qs_srt = qss;

#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (size_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = idx.indexes(qs, qs_srt, k);
      std::array<complex_t, dim> cache;
      for (size_t i = 0; i < dim; i++) {
        const auto ii = inds[i];
        cache[i] = qreg[ii];
        qreg[ii] = 0.;
      }
      for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
          qreg[inds[i]] += U(i, j) * cache[j];
    }
  }
}

//------------------------------------------------------------------------------
// 1-Qubit Ideal Gates
//------------------------------------------------------------------------------

void IdealBackend::qc_gate(const uint_t qubit, const double theta,
                           const double phi, const double lambda) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG qc_gate(" << qubit << ",{" << theta << "," << phi << ","
     << lambda << "})";
  std::clog << ss.str() << std::endl;
#endif
  qc_matrix1(qubit, waltz_matrix(theta, phi, lambda));
}

void IdealBackend::qc_gate_x(const uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_gate_x(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif

  // Optimized ideal Pauli-X gate
  const uint_t end2 = 1ULL << qubit; // end for k2 loop
  const uint_t step1 = end2 << 1;    // step for k1 loop
#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for collapse(2)
    for (uint_t k1 = 0; k1 < nstates; k1 += step1)
      for (uint_t k2 = 0; k2 < end2; k2++) {
        const auto i0 = k1 | k2;
        const auto i1 = i0 | end2;
        const complex_t cache = qreg[i0];
        qreg[i0] = qreg[i1]; // U(0,1)
        qreg[i1] = cache;    // U(1,0)
      }
  }
}

void IdealBackend::qc_gate_y(const uint_t qubit) {
// Optimized pauli Y gate
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_gate_y(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  // Optimized ideal Pauli-Y gate
  const uint_t end2 = 1ULL << qubit; // end for k2 loop
  const uint_t step1 = end2 << 1;    // step for k1 loop
  const complex_t I(0., 1.);
#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for collapse(2)
    for (uint_t k1 = 0; k1 < nstates; k1 += step1)
      for (uint_t k2 = 0; k2 < end2; k2++) {
        const auto i0 = k1 | k2;
        const auto i1 = i0 | end2;
        const complex_t cache = qreg[i0];
        qreg[i0] = -I * qreg[i1]; // U(0,1)
        qreg[i1] = I * cache;     // U(1,0)
      }
  }
}

void IdealBackend::qc_phase(const uint_t qubit, const complex_t phase) {
// optimized Z rotation (see useful_matrices.RZ)
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_phase(" << qubit << ", " << phase << ")";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end2 = 1ULL << qubit; // end for k2 loop
  const uint_t step1 = end2 << 1;    // step for k1 loop

#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for collapse(2)
    for (uint_t k1 = 0; k1 < nstates; k1 += step1)
      for (uint_t k2 = 0; k2 < end2; k2++) {
        const auto i1 = k1 | k2 | end2;
        qreg[i1] *= phase;
      }
  }
}

void IdealBackend::qc_zrot(const uint_t qubit, const double lambda) {
// optimized Z rotation
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_zrot(" << qubit << ",{" << lambda << "})";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end2 = 1ULL << qubit; // end for k2 loop
  const uint_t step1 = end2 << 1;    // step for k1 loop
  const complex_t phase = exp(complex_t(0, lambda));

#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for collapse(2)
    for (uint_t k1 = 0; k1 < nstates; k1 += step1)
      for (uint_t k2 = 0; k2 < end2; k2++) {
        const auto i1 = k1 | k2 | end2;
        qreg[i1] *= phase;
      }
  }
}

//------------------------------------------------------------------------------
// 2-Qubit Ideal Gates
//------------------------------------------------------------------------------

void IdealBackend::qc_cnot(const uint_t q_ctrl, const uint_t q_trgt) {
// optimized ideal CNOT on two qubits
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_cnot(" << q_ctrl << ", " << q_trgt << ")";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end = nstates >> 2;
  const auto qs_srt = (q_ctrl < q_trgt)
                          ? std::array<uint_t, 2>{{q_ctrl, q_trgt}}
                          : std::array<uint_t, 2>{{q_trgt, q_ctrl}};

#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (size_t k = 0; k < end; k++) {
      const auto i0 = idx.index0(qs_srt, k);
      const auto i1 = i0 | idx.bits[q_ctrl];
      const auto i3 = i1 | idx.bits[q_trgt];
      const complex_t cache = qreg[i3];
      qreg[i3] = qreg[i1];
      qreg[i1] = cache;
    }
  } // end omp parallel
}

void IdealBackend::qc_cz(const uint_t q_ctrl, const uint_t q_trgt) {
// optimized ideal CZ gate on two qubits
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_cz(" << q_ctrl << ", " << q_trgt << ")";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end = nstates >> 2;
  const auto qs_srt = (q_ctrl < q_trgt)
                          ? std::array<uint_t, 2>{{q_ctrl, q_trgt}}
                          : std::array<uint_t, 2>{{q_trgt, q_ctrl}};

#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (uint_t k = 0; k < end; k++) {
      const auto i0 = idx.index0(qs_srt, k);
      const auto i3 = i0 | idx.bits[q_ctrl] | idx.bits[q_trgt];
      qreg[i3] *= -1.;
    }
  }
}

void IdealBackend::qc_zzrot(const uint_t q0, const uint_t q1,
                            const double lambda) {
// optimized ZZ rotation (see useful_matrices.RZZ)
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_zzrot(" << q0 << ", " << q1 << ")";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end = nstates >> 2;
  const auto qs_srt = (q0 < q1) ? std::array<uint_t, 2>{{q0, q1}}
                                : std::array<uint_t, 2>{{q1, q0}};
  const complex_t phase = exp(complex_t(0, lambda / 2.));

#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (uint_t k = 0; k < end; k++) {
      const auto i0 = idx.index0(qs_srt, k);
      const auto i1 = i0 | idx.bits[q0];
      const auto i2 = i0 | idx.bits[q1];
      qreg[i1] *= phase;
      qreg[i2] *= phase;
    }
  }
}

//------------------------------------------------------------------------------
// Matrices
//------------------------------------------------------------------------------

cmatrix_t IdealBackend::waltz_matrix(double theta, double phi, double lambda) {
  const complex_t I(0., 1.);
  cmatrix_t U(2, 2);
  U(0, 0) = std::cos(theta / 2.);
  U(0, 1) = -std::exp(I * lambda) * std::sin(theta / 2.);
  U(1, 0) = std::exp(I * phi) * std::sin(theta / 2.);
  U(1, 1) = std::exp(I * (phi + lambda)) * std::cos(theta / 2.);
  return U;
}

//------------------------------------------------------------------------------
// Measurement
//------------------------------------------------------------------------------

void IdealBackend::qc_measure(const uint_t qubit, const uint_t cbit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_measure(" << qubit << "," << cbit << ")";
  std::clog << ss.str() << std::endl;
#endif

  // Actual measurement outcome
  const std::pair<uint_t, double> meas = qc_measure_outcome(qubit);
  creg[cbit] = meas.first; // Update register with noisy outcome
  qc_measure_reset(qubit, meas.first, meas);
}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------

void IdealBackend::qc_reset(const uint_t qubit, const uint_t state) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::reset(" << qubit << ", " << state << ")";
  std::clog << ss.str() << std::endl;
#endif
  // Simulate unobserved measurement
  qc_measure_reset(qubit, state, qc_measure_outcome(qubit));
}

//------------------------------------------------------------------------------
// Measurement Outcome and probability of outcome
//------------------------------------------------------------------------------

std::pair<uint_t, double> IdealBackend::qc_measure_outcome(const uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_measure_outcome(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end2 = 1ULL << qubit; // end for k2 loop
  const uint_t step1 = end2 << 1;    // step for k1 loop
  double p0 = 0.;
#pragma omp parallel reduction(+ : p0) if (omp_flag &&omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#pragma omp for collapse(2)
    for (uint_t k1 = 0; k1 < nstates; k1 += step1)
      for (uint_t k2 = 0; k2 < end2; k2++) {
        const auto v = qreg[k1 | k2];
        p0 += std::real(std::conj(v) * v);
      }
  }

  rvector_t probs = {p0, 1. - p0};

  const uint_t n = rng.rand_int(probs); // randomly pick outcome
  std::pair<uint_t, double> result(n, probs[n]);
  return result;
}

//------------------------------------------------------------------------------
// Post-Measurement Reset
//------------------------------------------------------------------------------
void IdealBackend::qc_measure_reset(const uint_t qubit,
                                    const uint_t reset_state,
                                    std::pair<uint_t, double> meas_result) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_meas_reset(" << qubit << ", " << reset_state
     << ", " << meas_result << ")";
  std::clog << ss.str() << std::endl;
#endif

  const uint_t end2 = 1ULL << qubit; // end for k2 loop
  const uint_t step1 = end2 << 1;    // step for k1 loop
  const double renorm = 1. / std::sqrt(meas_result.second);

  switch (reset_state) {

  case 0: {
    // Reset to |0> State
    if (meas_result.first == 0) { // Measurement outcome was 0
#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
      {
#pragma omp for collapse(2)
        for (uint_t k1 = 0; k1 < nstates; k1 += step1)
          for (uint_t k2 = 0; k2 < end2; k2++) {
            const auto i0 = k1 | k2;
            const auto i1 = i0 | end2;
            qreg[i0] *= renorm;
            qreg[i1] = 0.;
          }
      }
    } else { // Measurement outcome was 1
#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
      {
#pragma omp for collapse(2)
        for (uint_t k1 = 0; k1 < nstates; k1 += step1)
          for (uint_t k2 = 0; k2 < end2; k2++) {
            const auto i0 = k1 | k2;
            const auto i1 = i0 | end2;
            qreg[i0] = renorm * qreg[i1];
            qreg[i1] = 0;
          }
      }
    }
  } break;

  case 1: {
    // Reset to |1> State
    if (meas_result.first == 0) { // Measurement outcome was 0
#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
      {
#pragma omp for collapse(2)
        for (uint_t k1 = 0; k1 < nstates; k1 += step1)
          for (uint_t k2 = 0; k2 < end2; k2++) {
            const auto i0 = k1 | k2;
            const auto i1 = i0 | end2;
            qreg[i0] = renorm * qreg[i1];
            qreg[i1] = 0;
          }
      }
    } else { // Measurement outcome was 0
#pragma omp parallel if (omp_flag &&omp_threads > 1) num_threads(omp_threads)
      {
#pragma omp for collapse(2)
        for (uint_t k1 = 0; k1 < nstates; k1 += step1)
          for (uint_t k2 = 0; k2 < end2; k2++) {
            const auto i0 = k1 | k2;
            const auto i1 = i0 | end2;
            qreg[i0] = 0.;
            qreg[i1] *= renorm;
          }
      }
    }
  } break;

  default:
    std::stringstream msg;
    msg << "invalid reset state '" << reset_state << "'";
    throw std::runtime_error(msg.str());
  }
}

/*
// New multi-qubit measurement
// Should move this and index functions out of class into a separate library
template <size_t N>
std::discrete_distribution<>
IdealBackend::measure_probs(const std::array<uint_t, N> qs,
                            const cvector_t &state) {
  const size_t end = state.size() >> N;
  auto qs_srt = qs;
  std::sort(qs_srt.begin(), qs_srt.end());
  auto probs = std::array<double, 1ULL << N>(); // initialize to all 0s;
  // to add omp parallelization we would need to define a custom reduction
  for (size_t k = 0; k < end; k++) {
    const auto inds = index(qs, qs_srt, k);
    for (size_t i = 0; i < (1ULL << N); i++) {
      const auto q = state[inds[i]];
      probs[i] += std::real(std::conj(q) * q);
    }
  }
  std::discrete_distribution<> ret(probs.begin(), probs.end());
  return ret;
}
*/
//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif