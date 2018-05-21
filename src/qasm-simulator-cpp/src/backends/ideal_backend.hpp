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
#define _USE_MATH_DEFINES
#include <math.h>

#include "base_backend.hpp"
#include "qubit_vector.hpp"

namespace QISKIT {

/*******************************************************************************
 *
 * IdealBackend class
 *
 ******************************************************************************/

class IdealBackend : public BaseBackend<QubitVector> {

public:
  /************************
   * Constructors
   ************************/

  IdealBackend() = default;

  /************************
   * BaseBackend Methods
   ************************/
  void set_config(json_t &config) override;
  void initialize(const Circuit &prog) override;
  void qc_operation(const operation &op) override;

  /************************
   * GateSet
   ************************/

  const static gateset_t gateset;


protected:
  /************************
   * OpenMP setting
   ************************/
  int omp_threshold = 16;

  /************************
   * Apply matrices
   ************************/
  cvector_t vectorize_matrix(const cmatrix_t &mat) const;

  void qc_matrix1(const uint_t qubit, const cmatrix_t &U) {
    qreg.apply_matrix(qubit, vectorize_matrix(U));
  };
  void qc_matrix2(const uint_t qubit0, const uint_t qubit1, const cmatrix_t &U){
    qreg.apply_matrix<2>({{qubit0, qubit1}}, vectorize_matrix(U));
  };

  inline void qc_matrix1(const uint_t qubit, const cvector_t &U) {
    qreg.apply_matrix(qubit, U);
  };
  inline void qc_matrix2(const uint_t qubit0, const uint_t qubit1, const cvector_t &U) {
    qreg.apply_matrix<2>({{qubit0, qubit1}}, U);
  };

  /************************
   * Measurement and Reset
   ************************/

  virtual void qc_reset(const uint_t qubit, const uint_t state = 0);
  virtual void qc_measure(const uint_t qubit, const uint_t bit);
  virtual std::pair<uint_t, double> qc_measure_outcome(const uint_t qubit);

  /************************
   * 1-Qubit Gates
   ************************/
  virtual cmatrix_t waltz_matrix(const double theta, const double phi,
                                 const double lambda);
  virtual cvector_t waltz_vectorized_matrix(const double theta, const double phi,
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
 * BaseBackend methods
 *
 ******************************************************************************/

void IdealBackend::set_config(json_t &config) {
  // Set OMP threshold for state update functions
  JSON::get_value(omp_threshold, "threshold_omp_gate", config);
  // parse initial state from JSON
  if (JSON::check_key("initial_state", config)) {
    QubitVector initial_state = config["initial_state"].get<cvector_t>();
    bool renorm_initial_state = true;
    JSON::get_value(renorm_initial_state, "renorm", config);
    JSON::get_value(renorm_initial_state, "renorm_initial_state", config);
    if (renorm_initial_state)
      initial_state.renormalize();
      //renormalize(initial_state);
    if (initial_state.qubits() > 0)
      set_initial_state(initial_state);
  }
}

void IdealBackend::initialize(const Circuit &prog) {

  if (qreg_init_flag) {
    if (qreg_init.size() == (1ULL << prog.nqubits))
      // reset state to custom state
      qreg = qreg_init;
    else {
      std::string msg = "initial state is wong size for the circuit";
      throw std::runtime_error(msg);
    }
  } else {
    // reset state std::vector to default state
    qreg = QubitVector(prog.nqubits);
    qreg.set_omp_threshold(omp_threshold);
    qreg.set_omp_threads(num_threads);
    qreg.initialize();
  }

  // TODO: make a ClassicalRegister class using BinaryVector
  creg.assign(prog.nclbits, 0);
  qreg_saved.erase(qreg_saved.begin(), qreg_saved.end());
  qreg_snapshots.erase(qreg_snapshots.begin(), qreg_snapshots.end());
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
  case gate_t::RZZ:
    qc_zzrot(op.qubits[0], op.qubits[1], op.params[0]);
    break;
  case gate_t::Wait:
    break;
  // Commands
  case gate_t::Snapshot:
    snapshot_state(op.string_params[0]);
    break;
  case gate_t::Save:
    save_state(op.string_params[0]);
    break;
  case gate_t::Load:
    load_state(op.string_params[0]);
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
                                       // Waltz Gates
                                       {"u0", gate_t::U0},
                                       {"u1", gate_t::U1},
                                       {"u2", gate_t::U2},
                                       {"u3", gate_t::U3},
                                       // Two-qubit gates
                                       {"cx", gate_t::CX},
                                       {"cz", gate_t::CZ},
                                       {"rzz", gate_t::RZZ},
                                       // Simulator commands
                                       {"#wait", gate_t::Wait},
                                       {"#snapshot", gate_t::Snapshot},
                                       {"#noise", gate_t::Noise},
                                       {"#save", gate_t::Save},
                                       {"_load", gate_t::Load},
                                       {"wait", gate_t::Wait},
                                       {"snapshot", gate_t::Snapshot},
                                       {"noise", gate_t::Noise},
                                       {"save", gate_t::Save},          
                                       {"load", gate_t::Load}});

//------------------------------------------------------------------------------
// Unitary Matrices
//------------------------------------------------------------------------------

cvector_t IdealBackend::vectorize_matrix(const cmatrix_t &mat) const {
  // Assumes matrices are stored as column-major vectors
  cvector_t ret;
  ret.reserve(mat.size());
  for (size_t j=0; j < mat.size(); j++)
    ret.push_back(mat[j]);
  return ret;
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
  qreg.apply_matrix(qubit, waltz_vectorized_matrix(theta, phi, lambda));
}

void IdealBackend::qc_gate_x(const uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_gate_x(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg.apply_x(qubit);
}

void IdealBackend::qc_gate_y(const uint_t qubit) {
// Optimized pauli Y gate
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_gate_y(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg.apply_y(qubit);
}

void IdealBackend::qc_phase(const uint_t qubit, const complex_t phase) {
// optimized Z rotation (see useful_matrices.RZ)
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_phase(" << qubit << ", " << phase << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg.apply_matrix(qubit, cvector_t({1., phase}));
}

void IdealBackend::qc_zrot(const uint_t qubit, const double lambda) {
// optimized Z rotation
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_zrot(" << qubit << ",{" << lambda << "})";
  std::clog << ss.str() << std::endl;
#endif
  qreg.apply_matrix(qubit, cvector_t({1., exp(complex_t(0, lambda))}));
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
  qreg.apply_cnot(q_ctrl, q_trgt);
}

void IdealBackend::qc_cz(const uint_t q_ctrl, const uint_t q_trgt) {
// optimized ideal CZ gate on two qubits
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_cz(" << q_ctrl << ", " << q_trgt << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg.apply_cz(q_ctrl, q_trgt);
}

void IdealBackend::qc_zzrot(const uint_t q0, const uint_t q1,
                            const double lambda) {
// optimized ZZ rotation
// Has overall global phase set so that
// uzz(lambda) = exp(i*lambda/2) * exp(-I*lambda*(ZZ /2))
// OR equivalently uzz(lambda) q0, q1; = cx q0, q1; u1 q1; cx q0, q1;
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG IdealBackend::qc_zzrot(" << q0 << ", " << q1 << ")";
  std::clog << ss.str() << std::endl;
#endif
  const complex_t one(1.0, 0);
  const complex_t phase = exp(complex_t(0, lambda));
  qreg.apply_matrix<2>({{q0, q1}}, cvector_t({one, phase, phase, one}));
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

cvector_t IdealBackend::waltz_vectorized_matrix(double theta, double phi, double lambda) {
  const complex_t I(0., 1.);
  cvector_t U(4);
  U[0] = std::cos(theta / 2.);
  U[2] = -std::exp(I * lambda) * std::sin(theta / 2.);
  U[1] = std::exp(I * phi) * std::sin(theta / 2.);
  U[3] = std::exp(I * (phi + lambda)) * std::cos(theta / 2.);
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
  creg[cbit] = meas.first; // Update register outcome

  // Implement measurement                                            
  cvector_t mdiag(2, 0.);
  mdiag[meas.first] = 1. / std::sqrt(meas.second);
  qreg.apply_matrix(qubit, mdiag);
}

std::pair<uint_t, double> IdealBackend::qc_measure_outcome(const uint_t qubit) {

  // Probability of P0 outcome
  double p0 = qreg.probability(qubit, 0);
  rvector_t probs = {p0, 1. - p0};
  // randomly pick outcome
  const uint_t n = rng.rand_int(probs); 
  return std::pair<uint_t, double>(n, probs[n]);
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
  const std::pair<uint_t, double> meas = qc_measure_outcome(qubit);                                          
  cvector_t mdiag(2, 0.);
  mdiag[meas.first] = 1. / std::sqrt(meas.second);
  qreg.apply_matrix(qubit, mdiag);

  // if reset state disagrees with measurement outcome flip the qubit
  if (state != meas.first)
    qreg.apply_x(qubit);
}

//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif