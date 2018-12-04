/**
 * Copyright 2017, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    state_vector_engine.hpp
 * @brief   QISKIT Simulator QubitVector engine class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _VectorEngine_h_
#define _VectorEngine_h_

#include <sstream>
#include <stdexcept>

#include "qubit_vector.hpp"
#include "base_engine.hpp"
#include "misc.hpp"

namespace QISKIT {

/***************************************************************************/ /**
 *
 * VectorEngine class
 *
 * This class is derived from the BaseEngine class. It includes all options
 * and results of the BaseEngine, and in addition may compute properties
 * related to the state vector of the backend. These include:
 * - The density matrix of the snapshotted quantum states of the system system
 *   averaged over all shots
 * - The Z-basis measurement probabilities of the snapshotted quantum states of the
 *   system averaged over all shots. Note that this is equal to the diagonal of
 *   the density matrix.
 * - The ket representation of the snapshotted quantum states of the system for each
 *   shot
 * - The inner product with a set of target states of the snapshotted quantum states
 *   of the system for each shot.
 * - The expectation values of a set  of target states of the snapshotted quantum
 *   states of the system averaged over shots.
 *
 ******************************************************************************/

class VectorEngine : public BaseEngine<QubitVector> {

public:
  // Default constructor
  explicit VectorEngine(uint_t dim = 2)
      : BaseEngine<QubitVector>(), qudit_dim(dim){};
  ~VectorEngine() = default;
  //============================================================================
  // Configuration
  //============================================================================
  uint_t qudit_dim = 2;   // dimension of each qubit as qudit
  double epsilon = 1e-10; // Chop small numbers

  //============================================================================
  // Methods
  //============================================================================
  void execute(const Circuit &circ, BaseBackend<QubitVector> *be, uint_t nshots) override;

  // Adds results data from another engine.
  void add(const VectorEngine &eng);

  // Overloads the += operator to combine the results of different engines
  VectorEngine &operator+=(const VectorEngine &eng) {
    add(eng);
    return *this;
  };

  // Compute results
  template<class T>
  void sample_counts(const Circuit &prog, BaseBackend<QubitVector> *be, uint_t nshots,
                     const std::vector<T> &probs, const std::vector<operation> &meas,
                     const std::vector<uint_t> &meas_qubits);
};

/***************************************************************************/ /**
  *
  * VectorEngine methods
  *
  ******************************************************************************/

void VectorEngine::add(const VectorEngine &eng) {
  BaseEngine<QubitVector>::add(eng);
}

void VectorEngine::execute(const Circuit &prog, BaseBackend<QubitVector> *be,
                           uint_t nshots) {

  // Check to see if circuit is ideal and allows for measurement optimization
  if (prog.opt_meas && prog.noise.ideal) {

    // This optimization replaces the shots by a single shot + sampling
    // We need to subtract the additional shots added by BaseEngine class
    total_shots -= (nshots-1);

    // Find position of first measurement operation
    uint_t pos = 0;
    while (pos < prog.operations.size() &&
           prog.operations[pos].id != gate_t::Measure) {
      pos++;
    }
    // Execute operations before measurements
    std::vector<operation> not_meas(prog.operations.begin(),
                                    prog.operations.begin() + pos);
    be->initialize(prog);
    be->execute(not_meas);

    BaseEngine<QubitVector>::compute_results(prog, be);
    // Clear creg results from shot without measurements
    counts.clear();
    output_creg.clear();

    // Get measurement operations and set of measured qubits
    std::vector<operation> meas(prog.operations.begin() + pos,
                                prog.operations.end());
    std::vector<uint_t> meas_qubits;
    for (const auto &op : meas)
      meas_qubits.push_back(op.qubits[0]);

    // sort the qubits and delete duplicates
    sort(meas_qubits.begin(),meas_qubits.end());
    meas_qubits.erase(unique(meas_qubits.begin(), meas_qubits.end() ), meas_qubits.end());

    // Allow option to get probabilities in place be overwriting QubitVector
    bool probs_in_place = true;
    if (probs_in_place && meas_qubits.size() == prog.nqubits) {
      cvector_t &cprobs = be->access_qreg().vector();
      for (uint_t j=0; j < cprobs.size(); j++) {
        cprobs[j] = std::real(cprobs[j] * std::conj(cprobs[j]));
      }
      // Sample measurement outcomes
      sample_counts(prog, be, nshots, cprobs, meas, meas_qubits);
    } else {
      // Sample measurement outcomes
      rvector_t probs = be->access_qreg().probabilities(meas_qubits);
      sample_counts(prog, be, nshots, probs, meas, meas_qubits);
    }
  } else {
    // Standard execution of every shot
    BaseEngine<QubitVector>::execute(prog, be, nshots);
  }
}

// Templated so works for real or complex probability vector
template<class T>
void VectorEngine::sample_counts(const Circuit &prog, BaseBackend<QubitVector> *be, uint_t nshots,
                                 const std::vector<T> &probs, const std::vector<operation> &meas,
                                 const std::vector<uint_t> &meas_qubits) {
  // Map to store measured outcome after
  std::map<uint_t, uint_t> outcomes;
  for (auto &qubit : meas_qubits)
    outcomes[qubit] = 0;
  const uint_t N = meas_qubits.size(); // number of measured qubits

  // Sample measurement outcomes
  auto &rng = be->access_rng();
  for (uint_t shot = 0; shot < nshots; shot++) {
    double p = 0.;
    double r = rng.rand(0, 1);
    uint_t result;
    for (result = 0; result < (1ULL << N); result++) {
      if (r < (p += std::real(probs[result])))
        break;
    }
    // convert outcome to register
    auto reg = int2reg(result, 2, N);
    for (auto it = outcomes.cbegin(); it != outcomes.cend(); it++)
      outcomes[it->first] = reg[std::distance(outcomes.cbegin(), it)];
    // update creg
    auto &creg = be->access_creg();
    for (const auto &op : meas)
      creg[op.clbits[0]] = outcomes[op.qubits[0]];
    // compute count based results
    compute_counts(creg);
  }
}


/***************************************************************************/ /**
  *
  * JSON conversion
  *
  ******************************************************************************/

inline void to_json(json_t &js, const VectorEngine &eng) {

  // Get results from base class
  const BaseEngine<QubitVector> &base_eng = eng;
  to_json(js, base_eng);
}

inline void from_json(const json_t &js, VectorEngine &eng) {
  eng = VectorEngine();
  BaseEngine<QubitVector> &base_eng = eng;
  from_json(js, base_eng);

  // Get additional settings
  JSON::get_value(eng.epsilon, "chop", js);
  JSON::get_value(eng.qudit_dim, "qudit_dim", js);

  // renormalize state vector
  if (eng.initial_state_flag)
    eng.initial_state.renormalize();
}

//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif
