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
 * @file    base_backend.hpp
 * @brief   Base simualor backend for QISKIT simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _BaseBackend_h_
#define _BaseBackend_h_

#include <complex>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "circuit.hpp"
#include "noise_models.hpp"
#include "rng_engine.hpp"
#include "types.hpp"

namespace QISKIT {

/***************************************************************************/ /**
  *
  * BaseBackend class
  *
  * This is a base class for backends for the qasm_simulator. Additional
  *backends
  * must be derived from the BaseBackend type. Backends are defined by the Type
  * of their representation of the quantum state of the system, given by qreg,
  * and hence the BaseBackend is a template class based on this Type.
  *
  *
  ******************************************************************************/

template <class StateType> class BaseBackend {

public:
  /**
   * Default constructor for noiseless backend
   */
  BaseBackend() {
    ideal_sim = true;
    noise_flag = false;
    qreg_init_flag = false;
  };
  virtual ~BaseBackend() = default;

  /**
   * Execute a program on backend
   * @param prog
   */
  void execute(const Circuit &prog);
  void execute(const std::vector<operation> &ops);

  /**
   * Tests whether the the condition for implementing a conditional gate passes
   * @param
   * @return true if gate is to be implemented
   */
  bool qc_passed_if(const operation_if &cond);

  /**
   * Initialize backend
   * @param nq number of qubits
   * @param nc number of classical bits
   */
  virtual void initialize(const Circuit &prog) = 0;

  /**
   * Applies an operation to the backend state
   * @param op the operation to apply
   */
  virtual void qc_operation(const operation &op) = 0;

  /**
  * Sets the RNG seed of the backend to a fixed value.
  * @param seed: uint to use as RNG seed
  */
  inline void set_rng_seed(uint_t seed) { rng = RngEngine(seed); };

  /**
   * Set a custom initial state for the backend to initialize qreg to before
   * executing a QISKIT program
   * @param init the custom initial state
   */
  void set_initial_state(const StateType &init);

  // Templated state access methods

  /**
   * Returns a reference to the classical bit register of the backend
   * @return a reference to creg member
   */
  inline creg_t &access_creg() { return creg; };

  /**
   * Returns a reference to the jth bit of the classical bit register
   * @return a reference to creg[j]
   */
  inline uint_t &access_creg(uint_t j) { return creg[j]; };

  /**
   * Returns a reference to the backend RngEngine
   * @return a reference to rng
   */
  inline RngEngine &access_rng() { return rng; };

  /**
   * Returns a reference to the quantum register of the backend. The type
   * returned by this function is the template parameter of BaseBackend
   * @return a reference to qreg member
   */
  inline StateType &access_qreg() { return qreg; };

  /**
   * Returns a reference to the map of saved qreg states. These states are
   * chached during execution of a QISKIT program by the "save(j)" gate command.
   * The keys of the map are the integer arguments j.
   * @return a reference to saved qreg states
   */
  inline std::map<std::string, StateType> &access_saved() { return qreg_saved; };

  /**
   * Returns a reference to the map of qreg state snapshots. These states are
   * chached during execution of a QISKIT program by the "snapshot(j)" gate command.
   * The keys of the map are the integer arguments j.
   * @return a reference to qreg state snapshots
   */
  inline std::map<std::string, StateType> &access_snapshots() { return qreg_snapshots; };

  /**
   * Saves a copy of the current state of the qreg register to a map indexed by
   * the argument. If a saved state already exists for this key, the previous
   * value will be overwritten.
   * @param key the map key to save the state to
   */
  void save_state(std::string key);
  inline void save_state(double key) {
    save_state(std::to_string(static_cast<int>(key)));
  };
  /**
   * Saves a snapshot of the current state of the qreg register to a map indexed by
   * the argument. If a saved state already exists for this key, the previous
   * value will be overwritten.
   * @param key the map key to save the state to
   */
  void snapshot_state(std::string key);
  inline void snapshot_state(double key) {
    snapshot_state(std::to_string(static_cast<int>(key)));
  };
  /**
   * Loads a previously saved state of the system into qreg. If a saved state
   * does not exist for the argument key it will raise an error and terminate
   * the program.
   * @param key the map key to save the state to
   */
  void load_state(std::string key);
  inline void load_state(double key) {
    load_state(std::to_string(static_cast<int>(key)));
  };
  /**
   * Config Settings
   */

  virtual inline void set_config(json_t &config) = 0; // raises unused param warning
  inline void set_num_threads(int n) {
    if (n > 0)
      num_threads = n;
  };
  inline int get_num_threads() {
    return num_threads;
  };

  /**
   * Noise Settings
   */
  QubitNoise noise;

  void attach_noise(const QubitNoise &np);
  int_t reset_error(const uint_t state = 0);
  int_t measure_error(uint_t n);
  int_t relax_error();
  const GateError &gate_error(std::string gateName);

protected:
  /**
   * Number of threads to use for inner parallelization.
   */
  int num_threads = 1;
  
  /**
   * Stores the state of measured classical registers in a QISKIT program. All
   * bits are initialized in the 0 state, and any unmeasured bits will hence
   * return 0.
   */
  creg_t creg;

  /**
   * Stores a representation of the current quantum state of a backend as a
   * QISKIT
   * program is executed. Different backends may have different representations
   * of state, for example a complex vector, a density matrix, or Clifford
   * tableau. The Type for this representation is the template argument for the
   * BaseBackend class. This templating is necessary so that derrived classes
   * can return their state, no matter the type, using the 'access_qreg' method.
   */
  StateType qreg;

  /**
   * Stores the saved qreg states from the 'save_state' method. These saved
   * states can then be loaded as into qreg using the 'load_qreg' method.
   */
  std::map<std::string, StateType> qreg_saved;

  /**
   * Stores the saved qreg states from the 'save_state' method. These saved
   * states can then be loaded as into qreg using the 'load_qreg' method.
   */
  std::map<std::string, StateType> qreg_snapshots;

  /**
   * When set to 'true' this signals to the backend that when executing a QISKIT
   * program the qreg state should be initialized to a custom value, rather than
   * the default all zero state.
   */
  bool qreg_init_flag;

  /**
   * Stores the custom initial state that qreg will be initialized to if
   * qreg_init_flag is set to true.
   */
  StateType qreg_init;

  /**
   * RNG engine for backend. Used to generate random numbers for measurements
   * and noise processes
   */
  RngEngine rng;

  /**
   * When set to 'true' all operations should be implemented without noise
   */
  bool ideal_sim;

  /**
   * When set to 'true' signals to the backend to implement a noisy version of
   * the current QISKIT operation
   */
  bool noise_flag;
};

/*******************************************************************************
 *
 * Common methods for derived classes
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Applying operations
//------------------------------------------------------------------------------

template <class StateType>
void BaseBackend<StateType>::execute(const Circuit &prog) {
  // Initialize backend for circuit
  initialize(prog);

  // Run through operation list
  for (const auto &op : prog.operations)
    if (!op.if_op || (op.if_op && qc_passed_if(op.cond)))
      qc_operation(op);
}

template <class StateType>
void BaseBackend<StateType>::execute(const std::vector<operation> &ops) {
  // Run through operation list
  for (const auto &op : ops) {
    if (!op.if_op || (op.if_op && qc_passed_if(op.cond)))
      qc_operation(op);
  }
}

template <class StateType>
bool BaseBackend<StateType>::qc_passed_if(const operation_if &cond) {

  // Get masked vector
  creg_t masked;
  for (size_t j = 0; j < cond.mask.size(); j++)
    if (cond.mask[j] == 1)
      masked.push_back(creg[j]);

#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG BaseBackend::qc_passed_if(type=" << cond.type
     << ") = " << (cond.type == "equals" && cond.val == masked);
  std::cout << ss.str() << std::endl;
#endif
  return (cond.type == "equals" && cond.val == masked);
}

//------------------------------------------------------------------------------
// States
//------------------------------------------------------------------------------

template <class StateType>
void BaseBackend<StateType>::set_initial_state(const StateType &init) {
  qreg_init = init;
  qreg_init_flag = true;
}

template <class StateType> void BaseBackend<StateType>::save_state(std::string key) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG BaseBackend::save_state(" << key << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg_saved[key] = qreg;
}

template <class StateType> void BaseBackend<StateType>::snapshot_state(std::string key) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG BaseBackend::snapshot_state(" << key << ")";
  std::clog << ss.str() << std::endl;
#endif
  qreg_snapshots[key] = qreg;
}

template <class StateType> void BaseBackend<StateType>::load_state(std::string key) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG BaseBackend::load_state(" << key << ")";
  std::clog << ss.str() << std::endl;
#endif
  auto elem = qreg_saved.find(key);
  if (elem == qreg_saved.end()) {
    std::stringstream msg;
    msg << "could not load state, key \"" << key << "does not exist";
    throw std::runtime_error(msg.str());
  } else
    qreg = (elem->second);
}

//------------------------------------------------------------------------------
// Noise
//------------------------------------------------------------------------------

template <class StateType>
void BaseBackend<StateType>::attach_noise(const QubitNoise &np) {
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG: BaseBackend.attach_noise";
  std::clog << ss.str() << std::endl;
#endif
  // update noise params settings
  noise = np;
  ideal_sim = noise.ideal;
  noise_flag = !ideal_sim;
}

template <class StateType>
int_t BaseBackend<StateType>::reset_error(const uint_t state) {
  if (noise_flag && noise.reset.ideal == false) {
#ifdef DEBUG
    std::stringstream ss;
    ss << "DEBUG: reset_error(" << state << ")";
    std::clog << ss.str() << std::endl;
#endif
    return rng.rand_int(noise.reset.p);
  } else
    return state;
}

template <class StateType>
int_t BaseBackend<StateType>::measure_error(uint_t n) {
  if (noise_flag && noise.readout.ideal == false &&
      n < noise.readout.p.size()) {
#ifdef DEBUG
    std::stringstream ss;
    ss << "DEBUG: measure_error(" << n << ")";
    std::clog << ss.str() << std::endl;
#endif
    return rng.rand_int(noise.readout.p[n]);
  } else
    return n;
}

template <class StateType> int_t BaseBackend<StateType>::relax_error() {
  return rng.rand_int(noise.relax.populations);
}

template <class StateType>
const GateError &BaseBackend<StateType>::gate_error(std::string gateName) {
  return noise.gate[gateName];
}

//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif