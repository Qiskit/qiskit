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
 * @file    sampleshots_engine.hpp
 * @brief   optimized engine for ideal simulation with measurements in tail
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _SampleShotsEngine_h_
#define _SampleShotsEngine_h_

#include <sstream>
#include <stdexcept>

#include "ideal_backend.hpp"
#include "vector_engine.hpp"

namespace QISKIT {

/***************************************************************************/ /**
 *
 * SampleShotsEngine class
 *
 *
 ******************************************************************************/

class SampleShotsEngine : public VectorEngine {

public:
  // Default constructor
  SampleShotsEngine() : VectorEngine(2){};

  void execute(Circuit &prog, BaseBackend<QubitVector> *be, uint_t nshots);

protected:
};

/***************************************************************************/ /**
  *
  * SampleShotsEngine methods
  *
  ******************************************************************************/

void SampleShotsEngine::execute(Circuit &prog, BaseBackend<QubitVector> *be,
                                uint_t nshots) {
  if (prog.opt_meas) {
    // Find position of first measurement operation
    uint_t pos = 0;
    while (prog.operations[pos].id != gate_t::Measure &&
           pos < prog.operations.size()) {
      pos++;
    }
    // store measurements
    std::vector<operation> not_meas(prog.operations.begin(),
                                    prog.operations.begin() + pos);
    std::vector<operation> meas(prog.operations.begin() + pos,
                                prog.operations.end());
    
    be->initialize(prog);
    be->execute(not_meas);
  
    VectorEngine::compute_results(prog, be);
    // Clear creg results from shot without measurements
    counts.clear();
    output_creg.clear();

    // Get set of measured qubits
    std::vector<uint_t> meas_qubits;
    for (const auto &op : meas)
      meas_qubits.push_back(op.qubits[0]);
    // sort the qubits and delete duplicates
    sort(meas_qubits.begin(),meas_qubits.end()); 
    meas_qubits.erase(unique(meas_qubits.begin(), meas_qubits.end() ), meas_qubits.end());
    const uint_t N = meas_qubits.size(); // number of measured qubits

    auto &rng = be->access_rng();
    auto &creg = be->access_creg();
    auto &qreg = be->access_qreg();

    // get vector of outcome probabilities
    const rvector_t probs = qreg.probabilities(meas_qubits);

    // Map to store measured outcome after
    std::map<uint_t, uint_t> outcomes;
    for (auto &qubit : meas_qubits)
      outcomes[qubit] = 0;

    // sample measurement outcomes
    for (uint_t shot = 0; shot < nshots; shot++) {
      double p = 0.;
      double r = rng.rand(0, 1);
      uint_t result;
      for (result = 0; result < (1ULL << N); result++) {
        if (r < (p += probs[result]))
          break;
      }
      // convert outcome to register
      auto reg = int2reg(result, 2, N);
      for (auto it = outcomes.cbegin(); it != outcomes.cend(); it++)
        outcomes[it->first] = reg[std::distance(outcomes.cbegin(), it)];
      // update creg
      for (const auto &op : meas)
        creg[op.clbits[0]] = outcomes[op.qubits[0]];
      // compute count based results
      compute_counts(prog.clbit_labels, creg);
    }
  } else {
    // TODO -- maybe raise a runtime_error here
    // All measurements are not at the tail of circuit, so we do
    // standard VectorEngine execution
    VectorEngine::execute(prog, be, nshots);
  }
}

//------------------------------------------------------------------------------
} // end namespace QISKIT
#endif