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
 * quantum state, classical state, and saved quantum states, for each shot of
 * the simulation.
 *
 * This is done using default JSON conversion for the StateType, so if no such
 * conversion to JSON has been defined it will not be able to display the final
 * or saved quantum states.
 *
 * Derived Classes:
 *
 * Subclasses of the BaseEngine should make sure that overloads of the the `add`
 * `compute_results`, `to_json` and `from_json` functions call the BaseEngine
 * verions of those functions as well.
 ******************************************************************************/

template <typename StateType = cvector_t> class BaseEngine {

public:
  //============================================================================
  // Configuration
  //============================================================================

  bool counts_show = true;     // Dislay the map of final creg bitstrings
  bool counts_sort = true;     // Sort the map of bitstrings by occurence
  bool counts_space = true;    // Insert a space between named QISKIT cregs
  bool counts_bits_h2l = true; // Display bitstring with least sig to right

  bool show_final_creg = false; // Display a list of all observed outcomes
  bool show_final_qreg = false;
  bool show_saved_qreg = false;

  bool initial_state_flag = false;

  //============================================================================
  // Results / Data
  //============================================================================

  // Counts
  uint_t total_shots = 0; // Number of shots to obtain current results
  double time_taken = 0.; // Time taken for simulation of current results
  counts_t counts;        // Map of observed final creg values

  // Final States
  std::vector<std::string> output_creg; // creg string for each shot
  std::vector<StateType> output_qreg;   // final qreg state for each shot

  // Saved qreg states
  std::vector<std::map<uint_t, StateType>> saved_qreg;

  // Initial State
  StateType initial_state;

  // OMP Threshold for backend
  int_t omp_threshold = -1; // < 0 for automatic

  //============================================================================
  // Constructors
  //============================================================================

  BaseEngine(){};

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
  virtual void run_program(Circuit &circ, BaseBackend<StateType> *be,
                           uint_t nshots = 1, uint_t nthreads = 1);
  virtual void initialize(BaseBackend<StateType> *be, uint_t nthreads);
  virtual void execute(Circuit &circ, BaseBackend<StateType> *be,
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
  virtual void compute_results(Circuit &circ, BaseBackend<StateType> *be);

  void compute_counts(const reglist clbit_labels, const creg_t &creg);
};

/*******************************************************************************
 *
 * BaseEngine methods
 *
 ******************************************************************************/

template <typename StateType>
void BaseEngine<StateType>::run_program(Circuit &prog,
                                        BaseBackend<StateType> *be,
                                        uint_t nshots, uint_t nthreads) {
  initialize(be, nthreads);
  execute(prog, be, nshots);
  total_shots += nshots;
}

template <typename StateType>
void BaseEngine<StateType>::initialize(BaseBackend<StateType> *be,
                                       uint_t nthreads) {
  // Set OpenMP settings
  be->set_omp_threads(nthreads);
  be->set_omp_threshold(omp_threshold);
  // Set custom initial state
  if (initial_state_flag)
    be->set_initial_state(initial_state);
}

template <typename StateType>
void BaseEngine<StateType>::execute(Circuit &prog, BaseBackend<StateType> *be,
                                    uint_t nshots) {
  for (uint_t ishot = 0; ishot < nshots; ++ishot) {
    be->execute(prog);
    compute_results(prog, be);
  }
}

template <typename StateType>
void BaseEngine<StateType>::compute_results(Circuit &qasm,
                                            BaseBackend<StateType> *be) {
  // Compute counts
  compute_counts(qasm.clbit_labels, be->access_creg());

  // Final state
  if (show_final_qreg)
    output_qreg.push_back(be->access_qreg());

  // Saved states
  if (show_saved_qreg && be->access_saved().empty() == false)
    saved_qreg.push_back(be->access_saved());
}

template <typename StateType>
void BaseEngine<StateType>::compute_counts(const reglist clbit_labels,
                                           const creg_t &creg) {
  if (counts_show || show_final_creg) {
    std::string shotstr;
    uint_t shift = 0;

    for (const auto &reg : clbit_labels) {
      uint_t sz = reg.second;
      for (uint_t j = 0; j < sz; j++) {
        shotstr += std::to_string(creg[shift + j]);
      }
      shift += sz;
      if (counts_space)
        shotstr += " "; // opt whitespace between named cregs
    }
    if (shotstr.empty() == false && counts_space)
      shotstr.pop_back(); // remove last whitspace char

    // reverse shot string to least significant bit to the right
    if (counts_bits_h2l == true)
      std::reverse(shotstr.begin(), shotstr.end());

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
  // add time taken
  time_taken += eng.time_taken;

  // add total shots;
  total_shots += eng.total_shots;

  // copy counts
  for (auto pair : eng.counts)
    counts[pair.first] += pair.second;

  // copy output cregs
  std::copy(eng.output_creg.begin(), eng.output_creg.end(),
            std::back_inserter(output_creg));

  // copy output qregs
  std::copy(eng.output_qreg.begin(), eng.output_qreg.end(),
            back_inserter(output_qreg));

  // copy saved qregs
  std::copy(eng.saved_qreg.begin(), eng.saved_qreg.end(),
            back_inserter(saved_qreg));
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
    js["classical_states"] = engine.output_creg;

  if (engine.show_final_qreg && engine.output_qreg.empty() == false)
    try {
      // use try incase state class doesn't have json conversion method
      json_t js_qreg = engine.output_qreg;
      js["quantum_states"] = js_qreg;
    } catch (std::exception &e) {
      // Leave message in output that type conversion failed
      js["quantum_states"] = "Error: Failed to convert state type to JSON";
    }

  if (engine.show_saved_qreg && engine.saved_qreg.empty() == false)
    try {
      // use try incase state class doesn't have json conversion method
      json_t js_qreg = engine.saved_qreg;
      js["saved_quantum_states"] = js_qreg;
    } catch (std::exception &e) {
      // Leave message in output that type conversion failed
      js["saved_quantum_states"] =
          "Error: Failed to convert state type to JSON";
    }
}

template <typename StateType>
inline void from_json(const json_t &js, BaseEngine<StateType> &engine) {
  engine = BaseEngine<StateType>();
  std::vector<std::string> opts;
  if (JSON::get_value(opts, "data", js)) {
    for (auto &o : opts) {
      to_lowercase(o);
      string_trim(o);
      // check options
      if (o == "hidecounts")
        engine.counts_show = false;
      else if (o == "nospace")
        engine.counts_space = false;
      else if (o == "nosort")
        engine.counts_sort = false;
      else if (o == "reverse")
        engine.counts_bits_h2l = false;
      else if (o == "classicalstate" || o == "classicalstates")
        engine.show_final_creg = true;
      else if (o == "quantumstate" || o == "quantumstates")
        engine.show_final_qreg = true;
      else if (o == "savedquantumstates" || o == "savedquantumstate")
        engine.show_saved_qreg = true;
    }
  }

  // parse initial state from JSON
  if (JSON::get_value(engine.initial_state, "initial_state", js)) {
    engine.initial_state_flag = true;
  }

  // Get omp threshold
  JSON::get_value(engine.omp_threshold, "omp_threshold", js);
}

//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif