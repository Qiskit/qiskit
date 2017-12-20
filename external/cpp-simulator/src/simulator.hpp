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
 * @file    simulator.hpp
 * @brief   Simulator class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _Simulator_hpp_
#define _Simulator_hpp_

#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
// Parallelization
#ifdef _OPENMP
#include <omp.h> // OpenMP
#else
#include <future> // C++11
#endif

#include "circuit.hpp"
#include "misc.hpp"
#include "noise_models.hpp"
#include "types.hpp"

// Engines
#include "base_engine.hpp"
#include "sampleshots_engine.hpp"
#include "vector_engine.hpp"

// Backends
#include "clifford_backend.hpp"
#include "ideal_backend.hpp"
#include "qubit_backend.hpp"

namespace QISKIT {

/***************************************************************************/ /**
   *
   * Simulator class
   *
   ******************************************************************************/

class Simulator {
public:
  std::string id = "";             // simulation id
  std::string simulator = "qubit"; // simulator backend label
  std::vector<Circuit> circuits;   // QISKIT program

  // Multithreading Params
  uint_t max_memory_gb = 16;   // max memory to use
  uint_t max_threads_shot = 0; // 0 for automatic
  uint_t max_threads_gate = 0; // 0 for automatic

  // Constructor
  inline Simulator(){};

  // Execute all quantum circuits
  json_t execute();

  // Execute a single circuit
  template <class Engine, class Backend>
  json_t run_circuit(Circuit &circ) const;
};

/*******************************************************************************
 *
 * Simulator Methods
 *
 ******************************************************************************/

json_t Simulator::execute() {

  // Initialize ouput JSON
  std::chrono::time_point<myclock_t> start = myclock_t::now(); // start timer
  json_t ret;
  ret["id"] = id;
  if (simulator == "clifford")
    ret["backend"] = std::string("local_clifford_simulator");
  else
    ret["backend"] = std::string("local_qiskit_simulator");
  ret["simulator"] = simulator;

  // Choose simulator and execute circuits
  try {
    bool qobj_success = true;
    for (auto &circ : circuits) {
      json_t circ_res;

      // Choose Simulator Backend
      if (simulator == "clifford")
        circ_res = run_circuit<BaseEngine<Clifford>, CliffordBackend>(circ);
      else if (simulator == "ideal")
        circ_res = run_circuit<SampleShotsEngine, IdealBackend>(circ);
      else
        circ_res = run_circuit<VectorEngine, QubitBackend>(circ);

      // Check results
      qobj_success &= circ_res["success"].get<bool>();
      ret["result"].push_back(circ_res);
    }
    ret["time_taken"] =
        std::chrono::duration<double>(myclock_t::now() - start).count();
    ret["status"] = std::string("COMPLETED");
    ret["success"] = qobj_success;
  } catch (std::exception &e) {
    ret["success"] = false;
    ret["status"] = std::string("ERROR: ") + e.what();
  }
  return ret;
}

//------------------------------------------------------------------------------
template <class Engine, class Backend>
json_t Simulator::run_circuit(Circuit &circ) const {

  std::chrono::time_point<myclock_t> start = myclock_t::now(); // start timer
  json_t ret;                                                  // results JSON

  // Check max qubits
  uint_t max_qubits =
      static_cast<uint_t>(floor(log2(max_memory_gb * 1e9 / 16.)));
  if ((simulator == "qubit" || simulator == "ideal") &&
      circ.nqubits > max_qubits) {
    ret["success"] = false;
    std::stringstream msg;
    msg << "ERROR: Number of qubits (" << circ.nqubits
        << ") exceeds maximum memory (" << max_memory_gb << " GB).";
    ret["status"] = msg.str();
    return ret;
  }

  // Try to execute circuit
  try {
    // Initialize reference engine and backend from JSON config
    Engine engine = circ.config;
    Backend backend = circ.config;

    // Set RNG Seed
    uint_t rng_seed = (circ.rng_seed < 0) ? std::random_device()()
                                          : static_cast<uint_t>(circ.rng_seed);

// Thread number
#ifdef _OPENMP
    uint_t ncpus = omp_get_num_procs(); // OMP method
    omp_set_nested(1);                  // allow nested parallel threads
#else
    uint_t ncpus = std::thread::hardware_concurrency(); // C++11 method
#endif
    ncpus = std::max(1ULL, ncpus); // check 0 edge case
    int_t dq = (max_qubits > circ.nqubits) ? max_qubits - circ.nqubits : 0;
    uint_t threads = std::max<uint_t>(1UL, 2 * dq);
    if (simulator == "ideal" && circ.opt_meas)
      threads = 1; // single shot thread
    else {
      threads = std::min<uint_t>(threads, ncpus);
      threads = std::min<uint_t>(threads, circ.shots);
      if (max_threads_shot > 0)
        threads = std::min<uint_t>(max_threads_shot, threads);
    }
    uint_t gate_threads = std::max<uint_t>(1UL, ncpus / threads);
    if (max_threads_gate > 0)
      gate_threads = std::min<uint_t>(max_threads_gate, gate_threads);

    // Single-threaded shots loop
    if (threads < 2) {
      // Run shots on single-thread
      backend.set_rng_seed(rng_seed);
      engine.run_program(circ, &backend, circ.shots, gate_threads);
    }
    // Parallelized shots loop
    else {
      // Set rng seed for each thread
      std::vector<std::pair<uint_t, uint_t>> shotseed;
      for (uint_t j = 0; j < threads; ++j)
        shotseed.push_back(std::make_pair(circ.shots / threads, rng_seed + j));
      shotseed[0].first += (circ.shots % threads);

// OMP Execution
#ifdef _OPENMP
      std::vector<Engine> futures(threads);
#pragma omp parallel for if (threads > 1) num_threads(threads)
      for (uint_t j = 0; j < threads; j++) {
        const auto &ss = shotseed[j];
        Backend be(backend);
        be.set_rng_seed(ss.second);
        futures[j] = engine;
        futures[j].run_program(circ, &be, ss.first, gate_threads);
      }
      for (auto &f : futures)
        engine += f;

// C++11 Execution
#else
      std::vector<std::future<Engine>> futures;
      for (auto &&ss : shotseed)
        futures.push_back(async(std::launch::async, [&]() {
          Engine eng(engine);
          Backend be(backend);
          be.set_rng_seed(ss.second);
          eng.run_program(circ, &be, ss.first);
          return eng;
        }));
      // collect results
      for (auto &&f : futures)
        engine += f.get();
#endif
    } // end parallel shots

    // Return results
    ret["data"] = engine; // add engine output to return
    if (simulator != "ideal" && JSON::check_key("noise_params", circ.config)) {
      ret["noise_params"] = backend.noise;
    }

    // Add time taken and return result
    ret["data"]["time_taken"] =
        std::chrono::duration<double>(myclock_t::now() - start).count();
    // Add metadata
    ret["name"] = circ.name;
    ret["shots"] = circ.shots;
    ret["seed"] = rng_seed;
    if (threads > 1)
      ret["threads_shot"] = threads;
    // Report success
    ret["success"] = true;
    ret["status"] = std::string("DONE");
  } catch (std::exception &e) {
    ret["success"] = false;
    ret["status"] = std::string("ERROR: ") + e.what();
  }

  return ret;
}

//------------------------------------------------------------------------------
inline bool check_qobj(const json_t &qobj) {
  std::vector<std::string> qobj_keys{"id", "circuits"}; // optional: "config"
  std::vector<std::string> compiled_keys{"header", "operations"};
  std::vector<std::string> header_keys{"clbit_labels", "number_of_clbits",
                                       "number_of_qubits", "qubit_labels"};

  bool pass = JSON::check_keys(qobj_keys, qobj);
  if (pass) {
    for (auto &c : qobj["circuits"]) {
      pass &= JSON::check_key("compiled_circuit", c);
      if (pass) {
        pass &= JSON::check_keys(compiled_keys, c["compiled_circuit"]);
        if (pass)
          pass &=
              JSON::check_keys(header_keys, c["compiled_circuit"]["header"]);
      }
    }
  }
  return pass;
}

//------------------------------------------------------------------------------
inline void from_json(const json_t &js, Simulator &qobj) {
  try {
    if (check_qobj(js)) { // check valid qobj

      qobj = Simulator();
      JSON::get_value(qobj.id, "id", js);

      json_t config;
      JSON::get_value(config, "config", js);

      // Multithreading Parameters
      JSON::get_value(qobj.max_memory_gb, "max_memory", config);
      JSON::get_value(qobj.max_threads_shot, "max_threads_shot", config);
      JSON::get_value(qobj.max_threads_gate, "max_threads_gate", config);

      // Override with user simulator backend specification
      JSON::get_value(qobj.simulator, "simulator", config);
      to_lowercase(qobj.simulator);

      // Set simulator gateset
      gateset_t gateset;
      if (qobj.simulator == "qubit") {
        gateset = QubitBackend::gateset;
      } else if (qobj.simulator == "ideal") {
        gateset = IdealBackend::gateset;
      } else if (qobj.simulator == "clifford") {
        gateset = CliffordBackend::gateset;
      } else {
        throw std::runtime_error(std::string("invalid simulator."));
      }

      // Load unrolled qasm circuits
      const json_t &circs = js["circuits"];
      for (auto it = circs.cbegin(); it != circs.cend(); ++it)
        qobj.circuits.push_back(Circuit(*it, config, gateset));
    } else {
      throw std::runtime_error(std::string("invalid qobj file."));
    }
  } catch (std::exception &e) {
    std::stringstream msg;
    msg << "unable to parse qobj, " << e.what();
    throw std::runtime_error(msg.str());
  }
}

//------------------------------------------------------------------------------
} // end namespace QISKIT
//------------------------------------------------------------------------------
#endif