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

  void execute(Circuit &prog, BaseBackend<cvector_t> *be, uint_t nshots);

protected:
  /**
   * Converts a complex vector into the diagonal of the equivalent density
   * matrix by v[i] = v[i]*conj(v[i]) then performs a partial trace over
   * subsystems potentially reducing the dimension of vector.
   */
  void partial_trace(std::set<uint_t> &trsys, cvector_t &qreg);
};

/***************************************************************************/ /**
  *
  * SampleShotsEngine methods
  *
  ******************************************************************************/

void SampleShotsEngine::execute(Circuit &prog, BaseBackend<cvector_t> *be,
                                uint_t nshots) {
  if (prog.opt_meas) {
    // Find position of first measurement operation
    uint_t pos = 0;
    while (prog.operations[pos].id != gate_t::Measure &&
           pos < prog.operations.size()) {
      pos++;
    }
    // store measurements
    std::vector<operation> meas(prog.operations.begin() + pos,
                                prog.operations.end());
    // remove measurements from program and evaluate
    prog.operations.resize(pos);
    be->execute(prog); // execute gates without measurements
    // Note that calling compute results here will give probabilities,
    // state vectors, etc BEFORE measurement.
    VectorEngine::compute_results(prog, be);
    // Clear creg results from shot without measurements
    counts.clear();
    output_creg.clear();

    // Get set of measured qubits
    std::set<uint_t> qset;
    for (const auto &op : meas)
      qset.insert(op.qubits[0]);
    const uint_t N = qset.size(); // number of measured qubits

    // find set of qubits not measured
    std::set<uint_t> qtr;
    for (size_t j = 0; j < prog.nqubits; j++) {
      if (qset.find(j) == qset.end())
        qtr.insert(j);
    }

    auto &rng = be->access_rng();
    auto &creg = be->access_creg();
    auto &qreg = be->access_qreg();

    // trace over unmeasured qubits
    partial_trace(qtr, qreg);

    // container for meas outcomes indexed by measured qubit
    std::map<uint_t, uint_t> outcomes;
    for (auto &q : qset)
      outcomes[q] = 0;

    // sample measurement outcomes
    for (uint_t shot = 0; shot < nshots; shot++) {
      double p = 0.;
      double r = rng.rand(0, 1);
      uint_t result;
      for (result = 0; result < qreg.size(); result++) {
        if (r < (p += std::real(qreg[result])))
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
    // All measurements are not at the tail of circuit, so we do
    // standard VectorEngine execution
    VectorEngine::execute(prog, be, nshots);
  }
}

void SampleShotsEngine::partial_trace(std::set<uint_t> &trsys,
                                      cvector_t &qreg) {
  // Convert qreg to probability
  for (uint_t j = 0; j < qreg.size(); j++)
    qreg[j] *= std::conj(qreg[j]);

  const uint_t ntr = trsys.size();
  if (qreg.size() == 1ULL << ntr) {
    // trace all systems leaving scalar (length 1 vec)
    auto val = std::accumulate(qreg.begin(), qreg.end(), complex_t(0., 0.));
    qreg.resize(1);
    qreg[0] = val;
  } else if (ntr > 0) {
    // trace some subsystems
    MultiPartiteIndex idx;
    uint_t end = qreg.size();
    for (auto it = trsys.rbegin(); it != trsys.rend(); it++) {
      end >>= 1ULL;
      uint_t qubit = *it;
      for (size_t k = 0; k < end; k++) {
        const auto inds = idx.indexes<1>({{qubit}}, {{qubit}}, k);
        qreg[k] = qreg[inds[0]] + qreg[inds[1]];
      }
    }
    qreg.resize(end); // shrink vector
  }
}

//------------------------------------------------------------------------------
} // end namespace QISKIT
#endif