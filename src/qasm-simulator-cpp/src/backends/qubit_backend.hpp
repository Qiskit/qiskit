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
 * @file    qubit_backend.hpp
 * @brief   Standard Qubit QISKIT simulator backend
 * @authors Christopher J. Wood <cjwood@us.ibm.com>
 *          John A. Smolin <smolin@us.ibm.com>
 */

#ifndef _QubitBackend_hpp_
#define _QubitBackend_hpp_

#include <complex>
#include <string>
#include <vector>

#include "ideal_backend.hpp"

namespace QISKIT {

/*******************************************************************************
 *
 * QubitBackend class
 *
 ******************************************************************************/

class QubitBackend : public IdealBackend {

public:
  /************************
   * Constructors
   ************************/

  QubitBackend();

  /************************
   * BaseBackend Methods
   ************************/
  void initialize(const Circuit &prog) override;
  void qc_operation(const operation &op) override;

protected:
  /************************
   * Measurement and Reset
   ************************/

  void qc_reset(const uint_t qubit, const uint_t state = 0) override;
  void qc_measure(const uint_t qubit, const uint_t bit) override;

  /************************
   * 1-Qubit Gates
   ************************/

  // Single-qubit gates
  void qc_gate(const uint_t qubit, const double theta, const double phi,
                       const double lambda) override;
  void qc_idle(const uint_t qubit);
  void qc_gate_x(const uint_t qubit) override;
  void qc_gate_y(const uint_t qubit) override;

  void qc_u0(const uint_t qubit, const double n);
  void qc_u1(const uint_t qubit, const double lambda);
  void qc_u2(const uint_t qubit, const double phi, const double lambda);
  void qc_u3(const uint_t qubit, const double theta, const double phi,
             const double lambda);

  // 2-qubit gates
  void qc_cnot(const uint_t qctrl, const uint_t qtrgt) override;
  void qc_cz(const uint_t q0, const uint_t q1) override;

  // Gates with relaxation
  void qc_relax(const uint_t qubit, const double time);
  void qc_matrix1_noise(const uint_t qubit, const cmatrix_t &U,
                        const GateError &err);

  /************************
   * Matrices
   ************************/

  cmatrix_t pauli[4], pauli2[16];
  void add_pauli(const uint_t p, cmatrix_t &U);
  void add_pauli2(const uint_t p, cmatrix_t &U);

  virtual cmatrix_t rz_matrix(const double lambda);
  cmatrix_t noise_matrix1(const cmatrix_t &U, const GateError &err);
  cmatrix_t noise_matrix2(const cmatrix_t &U, const GateError &err);

  cmatrix_t U_X90_ideal;
  cmatrix_t U_CX_ideal;
  cmatrix_t U_CZ_ideal;
};


/*******************************************************************************
 *
 * BaseBackend methods
 *
 ******************************************************************************/

void QubitBackend::initialize(const Circuit &prog) {

  IdealBackend::initialize(prog);
  // system parameters
  noise_flag = !ideal_sim;
}

void QubitBackend::qc_operation(const operation &op) {

#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG: qc_operation";
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
  case gate_t::U0:
    qc_u0(op.qubits[0], op.params[0]);
    break; // u0 = wait(lambda * t_x90)
  case gate_t::U1:
    qc_u1(op.qubits[0], op.params[0]);
    break; // u1 = Rz(lambda) up to global phase
  case gate_t::U2:
    qc_u2(op.qubits[0], op.params[0], op.params[1]);
    break; // u2 = Rz(phi)*X90*Rz(lambda)
  case gate_t::U3:
    qc_u3(op.qubits[0], op.params[0], op.params[1], op.params[2]);
    break; // u3 = Rz(theta)*X90*Rz(phi)*X90*Rz(lambda)
  // QIP gates
  case gate_t::I:
    qc_idle(op.qubits[0]);
    break;
  case gate_t::X:
    qc_gate_x(op.qubits[0]);
    break;
  case gate_t::Y:
    qc_gate_y(op.qubits[0]);
    break;
  case gate_t::Z:
    qc_u1(op.qubits[0], M_PI);
    break;
  case gate_t::H:
    qc_u2(op.qubits[0], 0., M_PI);
    break;
  case gate_t::S:
    qc_u1(op.qubits[0], M_PI / 2.);
    break;
  case gate_t::Sd:
    qc_u1(op.qubits[0], -M_PI / 2.);
    break;
  case gate_t::T:
    qc_u1(op.qubits[0], M_PI / 4.);
    break;
  case gate_t::Td:
    qc_u1(op.qubits[0], -M_PI / 4.);
    break;
  case gate_t::CZ:
    qc_cz(op.qubits[0], op.qubits[1]);
    break;
  // ZZ rotation by angle lambda
  case gate_t::RZZ:
    IdealBackend::qc_zzrot(op.qubits[0], op.qubits[1], op.params[0]);
    break;
  case gate_t::Wait:
    if (noise_flag)
      qc_relax(op.qubits[0], op.params[0]);
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
    if (ideal_sim == false)
      noise_flag = (op.params[0] > 0.);
    break;
  // Invalid Gate (we shouldn't get here)
  default:
    std::string msg = "invalid QubitBackend operation";
    throw std::runtime_error(msg);
  }
}

/*******************************************************************************
 *
 * QubitBackend members
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------

QubitBackend::QubitBackend() : IdealBackend() {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::IdealBackend()";
  std::clog << ss.str() << std::endl;
#endif
  // Set OMP threshold
  omp_threshold = 20;
  // Set Pauli Matrices
  cmatrix_t id(2, 2), x(2, 2), y(2, 2), z(2, 2);
  MOs::Identity(id);
  MOs::Pauli(x, y, z);
  pauli[0] = id;
  pauli[1] = x;
  pauli[2] = y;
  pauli[3] = z;
  // set 2-qubit pauli matrices
  for (uint_t i = 0; i < 4; i++)
    for (uint_t j = 0; j < 4; j++)
      pauli2[i * 4 + j] = MOs::TensorProduct(pauli[i], pauli[j]);

  const complex_t I(0., 1.);

  U_CX_ideal.resize(4, 4);
  U_CX_ideal(0, 0) = 1.;
  U_CX_ideal(1, 3) = 1.;
  U_CX_ideal(2, 2) = 1.;
  U_CX_ideal(3, 1) = 1.;

  U_CZ_ideal.resize(4, 4);
  U_CZ_ideal(0, 0) = 1.;
  U_CZ_ideal(1, 1) = 1.;
  U_CZ_ideal(2, 2) = 1.;
  U_CZ_ideal(3, 3) = -1.;

  U_X90_ideal.resize(2, 2);
  U_X90_ideal(0, 0) = 1. / std::sqrt(2.);
  U_X90_ideal(0, 1) = -I / std::sqrt(2.);
  U_X90_ideal(1, 0) = -I / std::sqrt(2.);
  U_X90_ideal(1, 1) = 1. / std::sqrt(2.);
}

//------------------------------------------------------------------------------
// Measurement
//------------------------------------------------------------------------------

void QubitBackend::qc_measure(const uint_t qubit, const uint_t cbit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_measure(" << qubit << "," << cbit << ")";
  std::clog << ss.str() << std::endl;
#endif

  // Apply measurement noise gate

  if (noise_flag && gate_error("measure").ideal == false) {
    qc_matrix1_noise(qubit, pauli[0], gate_error("measure"));
  }

  // Compute measurement outcome
  const std::pair<uint_t, double> meas = qc_measure_outcome(qubit);
  // Update register with noisy outcome
  creg[cbit] = (noise_flag && noise.readout.ideal == false)
                   ? measure_error(meas.first)
                   : meas.first;

  // Implement measurement                                            
  cvector_t mdiag(2, 0.);
  mdiag[meas.first] = 1. / std::sqrt(meas.second);
  qreg.apply_matrix(qubit, mdiag);

}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------

void QubitBackend::qc_reset(const uint_t qubit, const uint_t state) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::reset(" << qubit << ", " << state << ")";
  std::clog << ss.str() << std::endl;
#endif

  // Apply reset with state reset error
  IdealBackend::qc_reset(qubit, reset_error(state));

  // Apply reset gate noise
  if (noise_flag && gate_error("reset").ideal == false) {
    qc_matrix1_noise(qubit, pauli[0], gate_error("reset"));
  }
}

//------------------------------------------------------------------------------
// 1-Qubit Gates
//------------------------------------------------------------------------------

void QubitBackend::qc_gate(const uint_t qubit, const double theta,
                           const double phi, const double lambda) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_gate(" << qubit << ",{" << theta << "," << phi
     << "," << lambda << "})";
  std::clog << ss.str() << std::endl;
#endif
  // Use "U" gate error
  if (noise_flag && gate_error("U").ideal == false) {
    qc_matrix1_noise(qubit, waltz_matrix(theta, phi, lambda),
                     gate_error("U")); // noisy gate
  }
  // Use "X90" gate error
  else if (noise_flag && gate_error("X90").ideal == false)
    qc_u3(qubit, theta, phi, lambda);
  // Ideal gate
  else
    // ideal single qubit unitary
    IdealBackend::qc_gate(qubit, theta, phi, lambda);
}

void QubitBackend::qc_u0(const uint_t qubit, const double n) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_u0(" << qubit << "," << n << ")";
  std::clog << ss.str() << std::endl;
#endif

  if (noise_flag && gate_error("X90").ideal == false) {
    // Use "X90" gate time
    qc_relax(qubit, n * gate_error("X90").gate_time);
  } else if (noise_flag && gate_error("U").ideal == false) {
    // Use U gate time if X90 is ideal
    qc_relax(qubit, n * gate_error("U").gate_time);
  }
}

void QubitBackend::qc_u1(const uint_t qubit, const double lambda) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_u1(" << qubit << ",{" << lambda << "})";
  std::clog << ss.str() << std::endl;
#endif
  // Use "U" gate error (if "X90" gate error is ideal)
  if (noise_flag && gate_error("X90").ideal && gate_error("U").ideal == false) {
    qc_matrix1_noise(qubit, waltz_matrix(0., 0., lambda), gate_error("U"));
  }
  // Ideal gate
  else
    IdealBackend::qc_zrot(qubit, lambda);
}

void QubitBackend::qc_u2(const uint_t qubit, const double phi,
                         const double lambda) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_u2(" << qubit << ",{" << phi << "," << lambda
     << "})";
  std::clog << ss.str() << std::endl;
#endif
  // Use "X90" gate error
  if (noise_flag && gate_error("X90").ideal == false) {
    // Noisy single qubit waltz-gate
    const GateError &err = gate_error("X90");
    cmatrix_t U = noise_matrix1(U_X90_ideal, err);
    U = rz_matrix(phi + M_PI / 2.) * U * rz_matrix(lambda - M_PI / 2.);
    qc_matrix1(qubit, U);
    qc_relax(qubit, err.gate_time);
  } else {
    cmatrix_t U = waltz_matrix(M_PI / 2., phi, lambda);
    // Use "U" gate error
    if (noise_flag && gate_error("U").ideal == false) {
      qc_matrix1_noise(qubit, U, gate_error("U"));
    }
    // Ideal u2 gate
    else
      qc_matrix1(qubit, U);
  }
}

void QubitBackend::qc_u3(const uint_t qubit, const double theta,
                         const double phi, const double lambda) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_u3(" << qubit << ",{" << theta << "," << phi
     << "," << lambda << "})";
  std::clog << ss.str() << std::endl;
#endif

  // Use "X90" gate error
  if (noise_flag && gate_error("X90").ideal == false) {
    qc_u2(qubit, theta / 2. + M_PI / 2., lambda + M_PI / 2.);
    qc_u2(qubit, phi + M_PI / 2., theta / 2. + M_PI / 2.);
  }
  // Use Single gate
  else {
    const cmatrix_t U = waltz_matrix(theta, phi, lambda);
    if (noise_flag && gate_error("U").ideal == false) {
      // Noisy single qubit waltz-gate
      qc_matrix1_noise(qubit, U, gate_error("U"));
    } else // ideal single qubit unitary
      qc_matrix1(qubit, U);
  }
}

void QubitBackend::qc_gate_x(const uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_gate_y(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  if (noise_flag && gate_error("X90").ideal == false) {
    // Noisy X90 based gate
    qc_u3(qubit, M_PI, 0., M_PI); // TODO update
  }
  // Use "U" gate error
  else if (noise_flag && gate_error("U").ideal == false) {
    qc_matrix1_noise(qubit, pauli[1], gate_error("U"));
  }
  // Optimized ideal Pauli-X gate
  else {
    IdealBackend::qc_gate_x(qubit);
  }
}

void QubitBackend::qc_gate_y(const uint_t qubit) {
// Optimized pauli Y gate
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_gate_y(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  // Use "X90" gate error
  if (noise_flag && gate_error("X90").ideal == false) {
    qc_u3(qubit, M_PI, M_PI / 2., M_PI / 2.); // TODO update
  }
  // Use "U" gate error
  else if (noise_flag && gate_error("U").ideal == false) {
    qc_matrix1_noise(qubit, pauli[2], gate_error("U"));
  }
  // Optimized ideal Pauli-Y gate
  else {
    IdealBackend::qc_gate_y(qubit);
  }
}

void QubitBackend::qc_idle(const uint_t qubit) {
  // Use "id" gate error
  if (noise_flag && gate_error("id").ideal == false) {
#ifdef DEBUG
    std::stringstream ss;
    ss << "DEBUG QubitBackend::qc_id(" << qubit << ")";
    std::clog << ss.str() << std::endl;
#endif
    qc_matrix1_noise(qubit, pauli[0], gate_error("id"));
  }
}

//------------------------------------------------------------------------------
// 2-Qubit Gates
//------------------------------------------------------------------------------

void QubitBackend::qc_cnot(const uint_t qubit_ctrl, const uint_t qubit_targ) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_cnot(" << qubit_ctrl << ", " << qubit_targ
     << ")";
  std::clog << ss.str() << std::endl;
#endif
  if (noise_flag && gate_error("CX").ideal == false) {
    // Apply noisy gate
    const GateError &err = gate_error("CX");
    const cmatrix_t U = noise_matrix2(U_CX_ideal, err);
    qc_matrix2(qubit_ctrl, qubit_targ, U);
    // apply relaxation to each qubit
    qc_relax(qubit_ctrl, err.gate_time);
    qc_relax(qubit_targ, err.gate_time);
  } else if (noise_flag && gate_error("CZ").ideal == false) {
    // implement as noisy CZ and hadamards
    qc_u2(qubit_targ, 0., M_PI);
    qc_cz(qubit_ctrl, qubit_targ);
    qc_u2(qubit_targ, 0., M_PI);
  } else
    IdealBackend::qc_cnot(qubit_ctrl, qubit_targ);
}

void QubitBackend::qc_cz(const uint_t qubit_ctrl, const uint_t qubit_targ) {
// optimized ideal CZ gate on two qubits
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG QubitBackend::qc_cz(" << qubit_ctrl << ", " << qubit_targ << ")";
  std::clog << ss.str() << std::endl;
#endif
  if (noise_flag && gate_error("CZ").ideal == false) {
    // Apply noisy gate
    const GateError &err = gate_error("CZ");
    const cmatrix_t U = noise_matrix2(U_CZ_ideal, err);
    qc_matrix2(qubit_ctrl, qubit_targ, U);
    // apply relaxation to each qubit
    qc_relax(qubit_ctrl, err.gate_time);
    qc_relax(qubit_targ, err.gate_time);
  } else if (noise_flag && gate_error("CX").ideal == false) {
    // implement as noisy CZ and hadamards
    qc_u2(qubit_targ, 0., M_PI);
    qc_cnot(qubit_ctrl, qubit_targ);
    qc_u2(qubit_targ, 0., M_PI);
  } else
    IdealBackend::qc_cz(qubit_ctrl, qubit_targ);
}

//------------------------------------------------------------------------------
// RZ-Matrix
//------------------------------------------------------------------------------
cmatrix_t QubitBackend::rz_matrix(const double lambda) {
  const complex_t I(0., 1.);
  cmatrix_t U(2, 2);
  U(0, 0) = 1.;
  U(1, 1) = std::exp(I * lambda);
  return U;
}

//------------------------------------------------------------------------------
// Add Noise to Matrices
//------------------------------------------------------------------------------

// multiply a unitary by a pauli matrix, doing nothing if j=0
void QubitBackend::add_pauli(const uint_t j, cmatrix_t &U) {
  if (j != 0 && j < 4)
    U = pauli[j] * U;
}

void QubitBackend::add_pauli2(const uint_t j, cmatrix_t &U) {
  if (j != 0 && j < 16)
    U = pauli2[j] * U;
}

cmatrix_t QubitBackend::noise_matrix1(const cmatrix_t &U,
                                      const GateError &err) {
  if (err.ideal) // If ideal return original matrix
    return U;
  // Add coherent error unitary
  cmatrix_t Uerr = (err.coherent_error) ? err.Uerr * U : U;
  // Add Pauli error
  add_pauli(rng.rand_int(err.pauli.p), Uerr);
  return Uerr;
}

cmatrix_t QubitBackend::noise_matrix2(const cmatrix_t &U,
                                      const GateError &err) {
  if (err.ideal) // If ideal return original matrix
    return U;
  // Add coherent error unitary
  cmatrix_t Uerr = (err.coherent_error) ? err.Uerr * U : U;
  // Add Pauli error
  add_pauli2(rng.rand_int(err.pauli.p), Uerr);
  return Uerr;
}

//------------------------------------------------------------------------------
// Noise Processes
//------------------------------------------------------------------------------

void QubitBackend::qc_relax(const uint_t qubit, const double time) {
  // applies relaxation to a qubit
  if (time > 0 && noise.relax.rate > 0) {
#ifdef DEBUG
    std::stringstream ss;
    ss << "DEBUG QubitBackend::qc_relax(" << qubit << ", t = " << time << ")";
    std::clog << ss.str() << std::endl;
#endif
    double p_relax = noise.relax.p(time);
    if (p_relax > 0. && rng.rand(0., 1.) < p_relax)
      IdealBackend::qc_reset(qubit, relax_error());
  }
}

void QubitBackend::qc_matrix1_noise(const uint_t qubit, const cmatrix_t &U,
                                    const GateError &err) {
// applyes unitary matrix with T1 relaxation error
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG: qc_matrix1_noise(" << qubit << ", err = " << err.label << ")";
  std::clog << ss.str() << std::endl;
#endif

  // Apply noisy gate
  qc_matrix1(qubit, noise_matrix1(U, err));
  // Apply relaxation
  qc_relax(qubit, err.gate_time);
}

//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif