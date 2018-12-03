/**
 * Copyright 2017, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    default_engine.hpp
 * @brief   QISKIT Simulator Engine base class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _BaseEngine_h_
#define _BaseEngine_h_

#include "base_backend.hpp"
#include "circuit.hpp"
#include "noise_models.hpp"

namespace QISKIT {

/***************************************************************************/ /**
 *
 * BaseEngine class
 *
 * This base class implements basic computation of simulator results for the
 * BaseBackend class and subclasses. This includes getting the bitstring values
 * from the classical register and basic functions for recording the final
 * quantum state, classical state, and snapshots of the quantum state, for each
 * shot of the simulation.
 *
 * This is done using default JSON conversion for the StateType, so if no such
 * conversion to JSON has been defined it will not be able to display the quantum
 * state snapshopts.
 *
 * Derived Classes:
 *
 * Subclasses of the BaseEngine should make sure that overloads of the the `add`
 * `compute_results`, `to_json` and `from_json` functions call the BaseEngine
 * versions of those functions as well.
 ******************************************************************************/

template <typename StateType = QubitVector> class BaseEngine {

public:
  //============================================================================
  // Configuration
  //============================================================================

  // Counts formatting

  bool counts_show = true;       // Dislay the map of final creg bitstrings
  bool hex_counts = true;        // Return counts as hexadecimal strings
  bool counts_sort = true;       // Sort the map of bitstrings by occurrence

  // Return list of final bitstrings for all shots
  bool show_final_creg = false; // Display a list of all observed outcomes

  // Return dictionary of snapshopts of quantum state
  bool show_snapshots = true;  // Display a list of qreg snapshots

  // Use a custom initial state for simulation
  bool initial_state_flag = false;

  //============================================================================
  // Results / Data
  //============================================================================

  // Counts
  uint_t total_shots = 0; // Number of shots to obtain current results
  double time_taken = 0.; // Time taken for simulation of current results
  counts_t counts;        // Map of observed final creg values

  // Quantum state snapshots
  std::map<std::string, std::vector<StateType>> snapshots;   // final qreg state for each shot

  // Classical states
  std::vector<std::string> output_creg; // creg string for each shot

  // Initial State
  StateType initial_state;

  //============================================================================
  // Constructors
  //============================================================================

  BaseEngine() = default;
  virtual ~BaseEngine() = default;

  //============================================================================
  // Methods
  //============================================================================

  /**
   * Sequentially executes a QISKIT program multiple times on a simulation
   * backend, extracts the value of the creg bitstrings after each simulation
   * shot and records them in the DefaultResults container. The results, and
   * format of results extracted is specified by the DefaultConfig object.
   * @param qasm the qasm_program to be executed on the backend
   * @param be a pointer to the backend to execute the qasm program on
   * @param nshots the number of simulation shots to run
   */
  virtual void run_program(const Circuit &circ, BaseBackend<StateType> *be,
                           uint_t nshots = 1);
  virtual void initialize(BaseBackend<StateType> *be);
  virtual void execute(const Circuit &circ, BaseBackend<StateType> *be,
                       uint_t nshots);

  /**
   * Adds results data from another engine.
   * @param eng the engine to combine.
   */
  void add(const BaseEngine<StateType> &eng);

  /**
   * Overloads the += operator to combine the results of different engines.
   */
  inline BaseEngine &operator+=(const BaseEngine<StateType> &eng) {
    add(eng);
    return *this;
  };

  /**
   * This function calculates results to based on the state of the backend after
   * the execution of each shot of a qasm program
   * @param qasm the qasm_program which was executed on the backend
   * @param be a pointer to the backend containing the state of the system
   *           after  execution of the qasm program
   */
  virtual void compute_results(const Circuit &circ, BaseBackend<StateType> *be);

  void compute_counts(const creg_t &creg);
};

/*******************************************************************************
 *
 * BaseEngine methods
 *
 ******************************************************************************/

template <typename StateType>
void BaseEngine<StateType>::run_program(const Circuit &prog,
                                        BaseBackend<StateType> *be,
                                        uint_t nshots) {
  initialize(be);
  execute(prog, be, nshots);
  total_shots += nshots;
}

template <typename StateType>
void BaseEngine<StateType>::initialize(BaseBackend<StateType> *be) {
  // Set custom initial state
  if (initial_state_flag)
    be->set_initial_state(initial_state);
}

template <typename StateType>
void BaseEngine<StateType>::execute(const Circuit &prog, BaseBackend<StateType> *be,
                                    uint_t nshots) {
  for (uint_t ishot = 0; ishot < nshots; ++ishot) {
    be->initialize(prog);
    be->execute(prog.operations);
    compute_results(prog, be);
  }
}

template <typename StateType>
void BaseEngine<StateType>::compute_results(const Circuit &qasm,
                                            BaseBackend<StateType> *be) {
  // Compute counts
  compute_counts(be->access_creg());

  // Snapshots
  if (show_snapshots && be->access_snapshots().empty() == false) {
    for (const auto& pair: be->access_snapshots()) {
      snapshots[pair.first].push_back(pair.second);
    }
  }
}

template <typename StateType>
void BaseEngine<StateType>::compute_counts(const creg_t &creg) {
  if (counts_show || show_final_creg) {

    // Convert reg to string
    // Note reg is in order [c0, c1, c2]
    // but bitstring is in order: c2c1c0
    std::string shotstr;
    for (auto it = creg.crbegin(); it != creg.crend(); ++it) {
      shotstr += std::to_string(*it);
    }

    // Convert to hexadecimal
    if (hex_counts)
      shotstr = bin2hex(shotstr, true);

    // add shot to shot map
    if (counts_show && shotstr.empty() == false)
      counts[shotstr] += 1;

    // add shot to shot history
    if (show_final_creg && shotstr.empty() == false)
      output_creg.push_back(shotstr);
  }
}

template <typename StateType>
void BaseEngine<StateType>::add(const BaseEngine<StateType> &eng) {
  time_taken += eng.time_taken;

  // add total shots;
  total_shots += eng.total_shots;

  // copy counts
  for (auto pair : eng.counts) {
    counts[pair.first] += pair.second;
  }

  // copy snapshots
  for (const auto &s: eng.snapshots) {
    std::copy(s.second.begin(), s.second.end(),
              back_inserter(snapshots[s.first]));
  }
  // copy output cregs
  std::copy(eng.output_creg.begin(), eng.output_creg.end(),
            std::back_inserter(output_creg));
}

/*******************************************************************************
 *
 * JSON conversion
 *
 ******************************************************************************/

template <typename StateType>
inline void to_json(json_t &js, const BaseEngine<StateType> &engine) {

  if (engine.counts_show && engine.counts.empty() == false)
    js["counts"] = engine.counts;

  if (engine.show_final_creg && engine.output_creg.empty() == false)
    js["memory"] = engine.output_creg;

  if (engine.show_snapshots && engine.snapshots.empty() == false) {
    try {
      // use try incase state class doesn't have json conversion method
      for (const auto& pair: engine.snapshots)
        js["snapshots"]["statevector"][pair.first] = pair.second;
    } catch (std::exception &e) {
      // Leave message in output that type conversion failed
      js["snapshots"]["statevector"] =
        "Error: Failed to convert state type to JSON";
    }
  }
  // check for edge case of null array and instead return empty object
  if (js.is_null())
    js = json_t::object();
}

template <typename StateType>
inline void from_json(const json_t &js, BaseEngine<StateType> &engine) {
  engine = BaseEngine<StateType>();
  // Check for single shot memory
  JSON::get_value(engine.show_final_creg, "memory", js);
  // parse initial state from JSON
  if (JSON::get_value(engine.initial_state, "initial_state", js)) {
    engine.initial_state_flag = true;
  }
}

//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif
