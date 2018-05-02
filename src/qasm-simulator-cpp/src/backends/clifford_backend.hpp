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
 * @file    clifford_backend.hpp
 * @brief   Clifford QISKIT simulator backend
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _CliffordBackend_
#define _CliffordBackend_

#include "base_backend.hpp"

namespace QISKIT {

/*******************************************************************************
 *
 * CliffordBackend
 *
 * This is base off the Clifford class defined in utilities/clifford.cpp
 *
 ******************************************************************************/

class CliffordBackend : public BaseBackend<Clifford> {

public:
  /************************
   * Constructors
   ************************/
  CliffordBackend() = default;

  /************************
   * BaseBackend Methods
   ************************/

  void set_config(json_t &config);
  void initialize(const Circuit &prog);
  void qc_operation(const operation &op);

  /************************
   * GateSet
   ************************/
  const static gateset_t gateset;

private:
  uint_t nqubits;

  /************************
   * Measurement and Reset
   ************************/

  void qc_reset(const uint_t qubit, const uint_t state = 0);
  void qc_measure(const uint_t qubit, const uint_t bit);

  /************************
   * Gates
   ************************/

  // Ideal single-qubit gates
  void qc_pauli(const uint_t qubit, const uint_t p); // ideal pauli

  // Noisy single qubit gates
  void qc_idle(const uint_t qubit);     // possibly noisy id
  void qc_gate_x(const uint_t qubit);   // possibly noisy x
  void qc_gate_y(const uint_t qubit);   // possibly noisy y
  void qc_gate_z(const uint_t qubit);   // possibly noisy z
  void qc_gate_h(const uint_t qubit);   // possibly noisy h
  void qc_gate_s(const uint_t qubit);   // possibly noisy s
  void qc_gate_sdg(const uint_t qubit); // possibly noisy sdg
  void qc_noise(const uint_t qubit, const GateError &gate,
                const bool X90 = false); // single qubit gate error

  // 2-qubit gates
  void qc_cnot(const uint_t qubit_c, const uint_t qubit_t);
  void qc_cz(const uint_t qubit_c, const uint_t qubit_t);
  // relaxation
  void qc_u0(const uint_t qubit, const double m);
  void qc_relax(const uint_t qubit, const double time); // relaxation error
  void qc_noise(const uint_t qubit_c, const uint_t qubit_t,
                const GateError &gate);
};

/*******************************************************************************
 *
 * BaseBackend methods
 *
 ******************************************************************************/

void CliffordBackend::set_config(json_t &config) {
  // parse initial state from JSON
  if (JSON::check_key("initial_state", config)) {
    Clifford initial_state = config["initial_state"];
    set_initial_state(initial_state);
  }
}

void CliffordBackend::initialize(const Circuit &prog) {

  nqubits = prog.nqubits;
  creg.assign(prog.nclbits, 0);
  noise_flag = !ideal_sim;

  qreg_saved.erase(qreg_saved.begin(), qreg_saved.end());
  qreg_snapshots.erase(qreg_snapshots.begin(), qreg_snapshots.end());

  if (qreg_init_flag) {
    if (qreg_init.size() == prog.nqubits)
      qreg = qreg_init;
    else {
      std::string msg = "initial state is wong size for the circuit";
      throw std::runtime_error(msg);
    }
  } else {
    qreg = Clifford(prog.nqubits);
  }
}

void CliffordBackend::qc_operation(const operation &op) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG: qc_operation";
  std::clog << ss.str() << std::endl;
#endif
  switch (op.id) {
  // Base gates
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
    qc_gate_z(op.qubits[0]);
    break;
  case gate_t::H:
    qc_gate_h(op.qubits[0]);
    break;
  case gate_t::S:
    qc_gate_s(op.qubits[0]);
    break;
  case gate_t::Sd:
    qc_gate_sdg(op.qubits[0]);
    break;
  case gate_t::CZ:
    qc_cz(op.qubits[0], op.qubits[1]);
    break;
  case gate_t::U0:
    qc_u0(op.qubits[0], op.params[0]);
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
    std::string msg = "invalid CliffordBackend operation";
    throw std::runtime_error(msg);
  }
}

/*******************************************************************************
 *
 * QubitBackend members
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Static member gateset
//------------------------------------------------------------------------------

const gateset_t CliffordBackend::gateset({// Core gates
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
                                          {"u0", gate_t::U0},
                                          {"sdg", gate_t::Sd},
                                          // Two-qubit gates
                                          {"cx", gate_t::CX},
                                          {"cz", gate_t::CZ},
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
// Measurement
//------------------------------------------------------------------------------

void CliffordBackend::qc_measure(const uint_t qubit, const uint_t cbit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_measure(" << qubit << "," << cbit << ")";
  std::clog << ss.str() << std::endl;
#endif

  // Check for noise and apply pre-measurement gate error
  if (noise_flag && gate_error("measure").ideal == false) {
    qc_noise(qubit, gate_error("measure"));
  }

  // randomly generate measurement outcome (even if deterministic)
  // this is to be consistant with rng for other engines
  const uint_t n = rng.rand_int(rvector_t({0.5, 0.5}));
  const uint_t meas = qreg.MeasZ(qubit, n); // Actual measurement outcome

  // Update register with noisy outcome
  creg[cbit] =
      (noise_flag && noise.readout.ideal == false) ? measure_error(meas) : meas;
}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------

void CliffordBackend::qc_reset(const uint_t qubit, const uint_t state) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_reset(" << qubit << "," << state << ")";
  std::clog << ss.str() << std::endl;
#endif

  // reset error state
  uint_t r = reset_error(state);
  // randomly generate measurement outcome (even if deterministic)
  // this is to be consistant with rng for other engines
  const uint_t n = rng.rand_int(rvector_t({0.5, 0.5}));
  qreg.PrepZ(qubit, n); // ideal reset to |0> state
  if (r == 1)
    qreg.X(qubit); // flip for error

  // Apply reset gate noise
  if (noise_flag && gate_error("reset").ideal == false)
    qc_noise(qubit, gate_error("reset"));
}

//------------------------------------------------------------------------------
// 1-Qubit Gates
//------------------------------------------------------------------------------

void CliffordBackend::qc_gate_h(uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_gate_h(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg.H(qubit); // ideal Hadamard

  if (noise_flag) {
    if (gate_error("gate").ideal == false)
      // H gate noise
      qc_noise(qubit, gate_error("gate"));
    else if (gate_error("X90").ideal == false)
      // X90 noise propagated through H gate
      qc_noise(qubit, gate_error("X90"), true);
  }
}

void CliffordBackend::qc_gate_s(uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_gate_s(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg.S(qubit); // ideal S

  if (noise_flag && gate_error("gate").ideal == false)
    // S gate noise
    qc_noise(qubit, gate_error("gate"));
}

void CliffordBackend::qc_gate_sdg(uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_gate_sdg(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg.Z(qubit);
  qreg.S(qubit); // ideal Sd

  if (noise_flag && gate_error("gate").ideal == false)
    // S gate noise
    qc_noise(qubit, gate_error("gate"));
}

void CliffordBackend::qc_gate_x(uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_gate_x(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif

  if (noise_flag && gate_error("gate").ideal == false) {
    // Noisy single qubit X gate
    qreg.X(qubit);
    qc_noise(qubit, gate_error("gate"));
  } else if (noise_flag && gate_error("X90").ideal == false) {
    // Noisy single qubit waltz-gate
    // X = H * Z * H
    qreg.H(qubit);
    qc_noise(qubit, gate_error("X90"), true);
    qreg.Z(qubit);
    qreg.H(qubit);
    qc_noise(qubit, gate_error("X90"), true);
  } else
    // ideal X gate
    qreg.X(qubit);
}

void CliffordBackend::qc_gate_y(uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_gate_y(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif

  if (noise_flag && gate_error("gate").ideal == false) {
    // Noisy single qubit Y gate
    qreg.Y(qubit);
    qc_noise(qubit, gate_error("gate"));
  } else if (noise_flag && gate_error("X90").ideal == false) {
    // Noisy single qubit waltz-gate
    // Y = S * X * S^dagger
    qreg.Z(qubit); // Sd = Z * S
    qreg.S(qubit);
    qreg.H(qubit); // H
    qc_noise(qubit, gate_error("X90"), true);
    qreg.Z(qubit); // Z
    qreg.H(qubit); // H
    qc_noise(qubit, gate_error("X90"), true);
    qreg.S(qubit); // S
  } else
    // ideal Y gate
    qreg.Y(qubit);
}

void CliffordBackend::qc_gate_z(uint_t qubit) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_gate_z(" << qubit << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg.Z(qubit); // ideal Z

  if (noise_flag && gate_error("gate").ideal == false)
    // Noisy single qubit gate
    qc_noise(qubit, gate_error("gate"));
}

void CliffordBackend::qc_idle(uint_t qubit) {

  if (noise_flag && !gate_error("id").ideal) {
#ifdef DEBUG
    std::stringstream ss;
    ss << "DEBUG CliffordBackend::qc_gate_id(" << qubit << ")";
    std::clog << ss.str() << std::endl;
#endif
    qc_relax(qubit, gate_error("id").gate_time);
  }
}

//------------------------------------------------------------------------------
// 2-Qubit Gates
//------------------------------------------------------------------------------

void CliffordBackend::qc_cnot(uint_t qubit_c, uint_t qubit_t) {

#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG ClifordBackend::qc_cnot(" << qubit_c << ", " << qubit_t << ")";
  std::clog << ss.str() << std::endl;
#endif
  if (noise_flag && gate_error("CX").ideal == false) {
    // Apply ideal gate
    qreg.CX(qubit_c, qubit_t);
    qc_noise(qubit_c, qubit_t, gate_error("CX"));
  } else if (noise_flag && gate_error("CZ").ideal == false) {
    // implement as noisy CZ and hadamards
    qc_gate_h(qubit_t);
    qc_cz(qubit_c, qubit_t);
    qc_gate_h(qubit_t);
  } else {
    // Ideal CX
    qreg.CX(qubit_c, qubit_t);
  }
}

void CliffordBackend::qc_cz(uint_t qubit_c, uint_t qubit_t) {

#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG ClifordBackend::qc_cz(" << qubit_c << ", " << qubit_t << ")";
  std::clog << ss.str() << std::endl;
#endif
  if (noise_flag && gate_error("CZ").ideal == false) {
    // Apply ideal gate
    qreg.CZ(qubit_c, qubit_t);
    qc_noise(qubit_c, qubit_t, gate_error("CZ"));
  } else if (noise_flag && gate_error("CX").ideal == false) {
    // implement as noisy CZ and hadamards
    qc_gate_h(qubit_t);
    qc_cnot(qubit_c, qubit_t);
    qc_gate_h(qubit_t);
  } else {
    // Ideal CX
    qreg.CZ(qubit_c, qubit_t);
  }
}

//------------------------------------------------------------------------------
// Ideal Pauli Operators
//------------------------------------------------------------------------------

void CliffordBackend::qc_pauli(const uint_t qubit, const uint_t p) {
  switch (p) {
  case 0: // id gate
    break;
  case 1: // X gate
    qreg.X(qubit);
    break;
  case 2: // Y gate
    qreg.Y(qubit);
    break;
  case 3: // Z gate
    qreg.Z(qubit);
    break;
  default:
    std::string msg = "pauli operator index out of bounds";
    throw std::runtime_error(msg);
  }
}

//------------------------------------------------------------------------------
// 1-Qubit Gate Noise
//------------------------------------------------------------------------------

/**
 * Suppse we assume gates are implmented as Waltz gates (ie QubitEngine)
 * then our error model is that errors occur on the non X90 gates.
 * To get this we rewrite H = S*X90*S.
 * Now the 3 pauli errors that can occur pass through S gate giving
 * S*X*X90*S = i*X*Z*H = Y*H
 * S*Y*X90*S = i*Y*Z*H = -X*H
 * S*Z*X90*S = Z*H
 */
void CliffordBackend::qc_noise(uint_t qubit, const GateError &gate, bool X90) {
#ifdef DEBUG
  std::cout << "DEBUG: qc_noise(" << qubit << ", " << gate.label << ")"
            << std::endl;
#endif

  // apply pauli channel error
  int_t err = rng.rand_int(gate.pauli.p);

  // Swap for hadamard gate error
  if (X90 && err == 1)
    err = 2;
  else if (X90 && err == 2)
    err = 1;
  qc_pauli(qubit, err);

  // relaxation
  qc_relax(qubit, gate.gate_time);
}

//------------------------------------------------------------------------------
// 2-Qubit Gate Noise
//------------------------------------------------------------------------------

void CliffordBackend::qc_noise(const uint_t qubit_c, const uint_t qubit_t,
                               const GateError &err) {
  // apply pauli channel error
  uint_t j = rng.rand_int(err.pauli.p);
  if (j > 0) {
    qc_pauli(qubit_c, j % 4);
    qc_pauli(qubit_t, j / 4);
  }
  // apply relaxation
  qc_relax(qubit_c, err.gate_time);
  qc_relax(qubit_t, err.gate_time);
}

//------------------------------------------------------------------------------
// Relaxation
//------------------------------------------------------------------------------

void CliffordBackend::qc_relax(const uint_t qubit, const double time) {
  if (time > 0 && noise.relax.rate > 0) {
#ifdef DEBUG
    std::stringstream ss;
    ss << "DEBUG CliffordBackend::qc_relax(" << qubit << ", t = " << time
       << ")";
    std::clog << ss.str() << std::endl;
#endif
    // applies relaxation to a qubit
    double p_relax = noise.relax.p(time);
    if (p_relax > 0. && rng.rand(0., 1.) < p_relax)
      qc_reset(qubit, relax_error());
  }
}

void CliffordBackend::qc_u0(const uint_t qubit, const double n) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG CliffordBackend::qc_u0(" << qubit << "," << n << ")";
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

//------------------------------------------------------------------------------
// end namespace QISKIT
}

#endif
