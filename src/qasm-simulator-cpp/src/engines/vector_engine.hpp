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
 * - The density matrix of the snapshoted quantum states of the system system
 *   averaged over all shots
 * - The Z-basis measurement probabilities of the snapshoted quantum states of the
 *   system averaged over all shots. Note that this is equal to the diagonal of
 *   the density matrix.
 * - The ket representation of the snapshoted quantum states of the sytem for each
 *   shot
 * - The inner product with a set of target states of the snapshoted quantum states
 *   of the system for each shot.
 * - The expectation values of a set  of target states of the snapshoted quantum 
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

  bool show_snapshots_ket = false;           // show snapshots as ket-vector
  bool show_snapshots_density = false;       // show snapshots as density matrix
  bool show_snapshots_probs = false;         // show snapshots as probability vector
  bool show_snapshots_probs_ket = false;     // show snapshots as probability ket-vector
  bool show_snapshots_inner_product = false; // show inner product with snapshots
  bool show_snapshots_overlaps = false;      // show overlaps with snapshots

  std::vector<QubitVector> target_states; // vector of target states

  //============================================================================
  // Results / Data
  //============================================================================

  // Snapshots output data
  std::map<std::string, std::vector<cket_t>> snapshots_ket;
  std::map<std::string, cmatrix_t> snapshots_density;
  std::map<std::string, rvector_t> snapshots_probs;
  std::map<std::string, std::map<std::string, double>> snapshots_probs_ket;
  std::map<std::string, std::vector<cvector_t>> snapshots_inprods;
  std::map<std::string, rvector_t> snapshots_overlaps;

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
  void compute_results(const Circuit &circ, BaseBackend<QubitVector> *be) override;
  template<class T>
  void sample_counts(const Circuit &prog, BaseBackend<QubitVector> *be, uint_t nshots,
                     const std::vector<T> &probs, const std::vector<operation> &meas,
                     const std::vector<uint_t> &meas_qubits);
  // Additional snapshot formatting
  void snapshot_ketform(const std::map<std::string, QubitVector>& qreg_snapshots,
                        const std::vector<uint_t> &labels);
  void snapshot_density_matrix(const std::map<std::string, QubitVector>& qreg_snapshots);
  void snapshot_probabilities(const std::map<std::string, QubitVector>& qreg_snapshots);
  void snapshot_inner_products(const std::map<std::string, QubitVector>& qreg_snapshots);

  // Convert a complex vector or ket to a real one
  double get_probs(const complex_t &val) const;
  rvector_t get_probs(const QubitVector &vec) const;
  std::map<std::string, double> get_probs(const cket_t &ket) const;
};

/***************************************************************************/ /**
  *
  * VectorEngine methods
  *
  ******************************************************************************/

void VectorEngine::add(const VectorEngine &eng) {

  BaseEngine<QubitVector>::add(eng);

  /* Accumulated snapshot sdata */

  // copy snapshots ket-maps
  for (const auto &s: eng.snapshots_ket)
    std::copy(s.second.begin(), s.second.end(),
              back_inserter(snapshots_ket[s.first]));

  // Add snapshots density
  for (const auto &s : eng.snapshots_density) {
    auto &rho = snapshots_density[s.first];
    if (rho.size() == 0)
      rho = s.second;
    else
      rho += s.second;
  }

  // Add snapshots probs
  for (const auto &s : eng.snapshots_probs) {
    snapshots_probs[s.first] += s.second;
  }

  // Add snapshots probs ket
  for (const auto &s : eng.snapshots_probs_ket)
    snapshots_probs_ket[s.first] += s.second;

  // Add snapshots overlaps
  for (const auto &s : eng.snapshots_overlaps)
    snapshots_overlaps[s.first] += s.second;

  // copy snapshots inner prods
  for (const auto &s : eng.snapshots_inprods)
    std::copy(s.second.begin(), s.second.end(),
              back_inserter(snapshots_inprods[s.first]));

}

void VectorEngine::execute(const Circuit &prog, BaseBackend<QubitVector> *be,
                           uint_t nshots) {

  // Check to see if circuit is ideal and allows for measurement optimization
  if (prog.opt_meas && prog.noise.ideal) {

    // Find position of first measurement operation
    uint_t pos = 0;
    while (prog.operations[pos].id != gate_t::Measure &&
           pos < prog.operations.size()) {
      pos++;
    }
    // Execute operations before measurements
    std::vector<operation> not_meas(prog.operations.begin(),
                                    prog.operations.begin() + pos);
    be->initialize(prog);
    be->execute(not_meas);

    VectorEngine::compute_results(prog, be);
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
    compute_counts(prog.clbit_labels, creg);
  }
}

//------------------------------------------------------------------------------

void VectorEngine::snapshot_ketform(const std::map<std::string, QubitVector>& qreg_snapshots,
                                    const std::vector<uint_t> &labels) {
  if (show_snapshots_ket || show_snapshots_probs_ket) {
    for (auto const &psi : qreg_snapshots) {
      cket_t psi_ket = vec2ket(psi.second.vector(), qudit_dim, epsilon, labels);
      // snapshots kets
      if (show_snapshots_ket)
        snapshots_ket[psi.first].push_back(psi_ket);
      // snapshots probabilities (ket form)
      if (show_snapshots_probs_ket) {
        rket_t probs_ket;
        for (const auto &val : psi_ket) {
          probs_ket[val.first] = get_probs(val.second);
        snapshots_probs_ket[psi.first] += probs_ket;
        }
      }
    }
  }
}

void VectorEngine::snapshot_density_matrix(const std::map<std::string, QubitVector>& qreg_snapshots) {
  if (show_snapshots_density) {
    for (auto const &psi : qreg_snapshots) {
      cmatrix_t &rho = snapshots_density[psi.first];
      if (rho.size() == 0)
        rho = outer_product(psi.second.vector(), psi.second.vector());
      else
        rho = rho + outer_product(psi.second.vector(), psi.second.vector());
    }
  }
}

void VectorEngine::snapshot_probabilities(const std::map<std::string, QubitVector>& qreg_snapshots) {
  if (show_snapshots_probs) {
      for (auto const &psi : qreg_snapshots) {
        auto &pr = snapshots_probs[psi.first];
        if (pr.empty())
          pr = get_probs(psi.second);
        else
          pr += get_probs(psi.second);
      }
    }
}

void VectorEngine::snapshot_inner_products(const std::map<std::string, QubitVector>& qreg_snapshots) {
  if (target_states.empty() == false &&
        (show_snapshots_inner_product || show_snapshots_overlaps)) {
      for (auto const &psi : qreg_snapshots) {
        // compute inner products
        cvector_t inprods;
        for (auto const &vec : target_states) {
          // check correct size
          if (vec.size() != psi.second.size()) {
            std::stringstream msg;
            msg << "error: target_state vector size \"" << vec.size()
                << "\" should be \"" << psi.second.size() << "\"";
            throw std::runtime_error(msg.str());
          }
          complex_t val = psi.second.inner_product(vec);
          chop(val, epsilon);
          inprods.push_back(val);
        }

        // add output inner products
        if (show_snapshots_inner_product)
          snapshots_inprods[psi.first].push_back(inprods);
        // Add output overlaps (needs renormalizing at output)
        if (show_snapshots_overlaps)
          snapshots_overlaps[psi.first] += get_probs(inprods);
      }
    }
}

void VectorEngine::compute_results(const Circuit &qasm, BaseBackend<QubitVector> *be) {
  // Run BaseEngine Counts
  BaseEngine<QubitVector>::compute_results(qasm, be);

  std::map<std::string, QubitVector> &qreg_snapshots = be->access_snapshots();

  /* Snapshot quantum state output data */
  if (snapshots.empty() == false) {

    // String labels for ket form
    std::vector<uint_t> ket_regs;
    for (auto it = qasm.qubit_sizes.crbegin(); it != qasm.qubit_sizes.crend(); ++it)
      ket_regs.push_back(it->second);
    // Snapshot ket-form of state vector or probabilities
    snapshot_ketform(qreg_snapshots, ket_regs);

    // add density matrix (needs renormalizing after all shots)
    snapshot_density_matrix(qreg_snapshots);

    // add probs (needs renormalizing after all shots)
    snapshot_probabilities(qreg_snapshots);

    // Inner products
    snapshot_inner_products(qreg_snapshots);
  }
}

//------------------------------------------------------------------------------
double VectorEngine::get_probs(const complex_t &val) const {
  return std::real(std::conj(val) * val);
}

rvector_t VectorEngine::get_probs(const QubitVector &vec) const {
  rvector_t ret;
  for (const auto &elt : vec.vector())
    ret.push_back(get_probs(elt));
  return ret;
}

std::map<std::string, double> VectorEngine::get_probs(const cket_t &ket) const {
  std::map<std::string, double> ret;
  for (const auto &elt : ket)
    ret[elt.first] = get_probs(elt.second);
  return ret;
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

  // renormalization constant for average over shots
  double renorm = 1. / eng.total_shots;

  /* Additional snapshot output data */
  // Snapshot skets
  if (eng.show_snapshots_ket && eng.snapshots_ket.empty() == false) {
    for (const auto& s: eng.snapshots_ket)
        js["snapshots"][s.first]["quantum_state_ket"] = s.second;
  }

  // Snapshots density
  if (eng.show_snapshots_density && eng.snapshots_density.empty() == false) {
    for (const auto &s : eng.snapshots_density) {
      auto rho = s.second * renorm;
      chop(rho, eng.epsilon);
      js["snapshots"][s.first]["density_matrix"] = rho;
    }
  }

  // Snapshots probs
  if (eng.show_snapshots_probs && eng.snapshots_probs.empty() == false) {
    for (const auto &s : eng.snapshots_probs) {
      auto val = s.second;
      val *= renorm;
      chop(val, eng.epsilon);
      js["snapshots"][s.first]["probabilities"] = val;
    }
  }

  // Snapshots probs ket
  if (eng.show_snapshots_probs_ket && eng.snapshots_probs_ket.empty() == false) {
    for (const auto &s : eng.snapshots_probs_ket) {
      auto val = s.second;
      val *= renorm;
      chop(val, eng.epsilon);
      js["snapshots"][s.first]["probabilities_ket"] = val;
    }
  }


  // Snapshots inner products
  if (eng.show_snapshots_inner_product && eng.snapshots_inprods.empty() == false) {
    for (const auto &s : eng.snapshots_inprods) {
      auto val = s.second;
      chop(val, eng.epsilon);
      js["snapshots"][s.first]["target_states_inner_product"] = val;
    }
  }

  // Snapshots overlaps
  if (eng.show_snapshots_overlaps && eng.snapshots_overlaps.empty() == false) {
    for (const auto &s : eng.snapshots_overlaps) {
      auto val = s.second;
      val *= renorm;
      chop(val, eng.epsilon);
      js["snapshots"][s.first]["target_states_inner_overlaps"] = val;
    }
  }
}

inline void from_json(const json_t &js, VectorEngine &eng) {
  eng = VectorEngine();
  BaseEngine<QubitVector> &base_eng = eng;
  from_json(js, base_eng);
  // Get output options
  std::vector<std::string> opts;
  if (JSON::get_value(opts, "data", js)) {
    for (auto &o : opts) {
      to_lowercase(o);
      string_trim(o);

      if (o == "quantumstateket" || o == "quantumstatesket")
        eng.show_snapshots_ket = true;
      else if (o == "densitymatrix")
        eng.show_snapshots_density = true;
      else if (o == "probabilities" || o == "probs")
        eng.show_snapshots_probs = true;
      else if (o == "probabilitiesket" || o == "probsket")
        eng.show_snapshots_probs_ket = true;
      else if (o == "targetstatesinnerproduct")
        eng.show_snapshots_inner_product = true;
      else if (o == "targetstatesoverlaps")
        eng.show_snapshots_overlaps = true;
    }
  }
  // Get additional settings
  JSON::get_value(eng.epsilon, "chop", js);
  JSON::get_value(eng.qudit_dim, "qudit_dim", js);

  // renormalize state vector
  if (eng.initial_state_flag)
    eng.initial_state.renormalize();

  // parse target states from JSON
  bool renorm_target_states = true;
  JSON::get_value(renorm_target_states, "renorm_target_states", js);
  if (JSON::get_value(eng.target_states, "target_states", js) &&
      renorm_target_states)
    for (auto &qv : eng.target_states)
      qv.renormalize();
}

//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif