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
 * @file    vector_engine.hpp
 * @brief   QISKIT Simulator State std::vector engine class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _VectorEngine_h_
#define _VectorEngine_h_

#include <sstream>
#include <stdexcept>

//#include "state_results.hpp"
#include "base_engine.hpp"
#include "misc.hpp"

// SAVED PROBABILITIES BROKEN
// SAVED PROB KET BROKEN

namespace QISKIT {

/***************************************************************************/ /**
 *
 * VectorEngine class
 *
 * This class is derived from the BaseEngine class. It includes all options
 * and results of the BaseEngine, and in addition may compute properties
 * related to the state vector of the backend. These include:
 * - The density matrix of the final or saved states of the system system
 *   averaged over all shots
 * - The Z-basis measurement probabilities of the final or saved states of the
 *   system averaged over all shots. Note that this is equal to the diagonal of
 *   the density matrix.
 * - The ket representation of the final or saved states of the sytem for each
 *   shot
 * - The inner product with a set of target states of the final or saved states
 *   of the system for each shot.
 * - The expectation values of a set  of target states of the final or saved
 *   states of the system averaged over shots.
 *
 ******************************************************************************/

class VectorEngine : public BaseEngine<cvector_t> {

public:
  // Default constructor
  VectorEngine(uint_t dim = 2) : BaseEngine<cvector_t>(), qudit_dim(dim){};

  //============================================================================
  // Configuration
  //============================================================================
  uint_t qudit_dim = 2;   // dimension of each qubit as qudit
  double epsilon = 1e-10; // Chop small numbers

  bool show_final_ket = false;       // return final state vectors
  bool show_final_density = false;   // return final density matrix of all shots
  bool show_final_probs = false;     // return final state probs
  bool show_final_probs_ket = false; // return final state probs
  bool show_final_inner_product = false; // compute ip with targ states
  bool show_final_overlaps = false;      // compute overlaps with target states

  bool show_saved_ket = false;           // record saved states
  bool show_saved_density = false;       // record saved density matrices
  bool show_saved_probs = false;         // record saved density matrices
  bool show_saved_probs_ket = false;     // record saved density matrices
  bool show_saved_inner_product = false; // compute ip with targ states
  bool show_saved_overlaps = false;      // compute overlaps with target states

  std::vector<cvector_t> target_states; // vector of target states

  //============================================================================
  // Results / Data
  //============================================================================

  // Final state output data
  std::vector<cket_t> output_ket; // quantum state ket
  cmatrix_t output_density;       // density matrix over all shots
  rvector_t output_probs;         // probability vec over all shots
  std::map<std::string, double>
      output_probs_ket; // probability ket over all shots

  std::vector<cvector_t> output_inprods; // inner products with target states
  rvector_t output_overlaps;             // average overlaps with target states

  // Saved states output data
  std::vector<std::map<uint_t, cket_t>> saved_ket;
  std::map<uint_t, cmatrix_t> saved_density;
  std::map<uint_t, rvector_t> saved_probs;
  std::map<uint_t, std::map<std::string, double>> saved_probs_ket;
  std::map<uint_t, std::vector<cvector_t>> saved_inprods;
  std::map<uint_t, rvector_t> saved_overlaps;

  //============================================================================
  // Methods
  //============================================================================

  // Adds results data from another engine.
  void add(const VectorEngine &eng);

  // Overloads the += operator to combine the results of different engines
  VectorEngine &operator+=(const VectorEngine &eng) {
    add(eng);
    return *this;
  };

  // Compute results
  void compute_results(Circuit &circ, BaseBackend<cvector_t> *be);

  // Convert a complex vector or ket to a real one
  double get_probs(const complex_t &val) const;
  rvector_t get_probs(const cvector_t &vec) const;
  std::map<std::string, double> get_probs(const cket_t &ket) const;
};

/***************************************************************************/ /**
  *
  * VectorEngine methods
  *
  ******************************************************************************/

void VectorEngine::add(const VectorEngine &eng) {

  BaseEngine<cvector_t>::add(eng);

  /* Accumulated output state data */

  // copy output ket-maps
  std::copy(eng.output_ket.begin(), eng.output_ket.end(),
            back_inserter(output_ket));

  // copy inner products
  std::copy(eng.output_inprods.begin(), eng.output_inprods.end(),
            back_inserter(output_inprods));

  // Add overlaps
  output_overlaps += eng.output_overlaps;

  // Add output probs ket (not normalized)
  output_probs_ket += eng.output_probs_ket;

  // Add probs (not normalized)
  output_probs += eng.output_probs;

  // Add density matrices (not normalized)
  if (output_density.size() == 0)
    output_density = eng.output_density;
  else
    output_density += eng.output_density;

  /* Accumulated saved states Data */

  // copy saved ket-maps
  std::copy(eng.saved_ket.begin(), eng.saved_ket.end(),
            back_inserter(saved_ket));

  // Add saved density
  for (const auto &save : eng.saved_density) {
    auto &rho = saved_density[save.first];
    if (rho.size() == 0)
      rho = save.second;
    else
      rho += save.second;
  }

  // Add saved probs
  for (const auto &save : eng.saved_probs) {
    saved_probs[save.first] += save.second;
  }

  // Add saved probs ket
  for (const auto &save : eng.saved_probs_ket)
    saved_probs_ket[save.first] += save.second;

  // Add saved overlaps
  for (const auto &save : eng.saved_overlaps)
    saved_overlaps[save.first] += save.second;

  // copy saved inner prods
  for (const auto &save : eng.saved_inprods)
    std::copy(save.second.begin(), save.second.end(),
              back_inserter(saved_inprods[save.first]));
}

//------------------------------------------------------------------------------

void VectorEngine::compute_results(Circuit &qasm, BaseBackend<cvector_t> *be) {
  // Run BaseEngine Counts
  BaseEngine<cvector_t>::compute_results(qasm, be);

  cvector_t &qreg = be->access_qreg();
  std::map<uint_t, cvector_t> &qreg_saved = be->access_saved();

  // String labels for ket form
  bool ket_form = (show_final_ket || show_saved_ket || show_final_probs_ket ||
                   show_saved_probs_ket);
  std::vector<uint_t> regs;
  if (ket_form)
    for (auto it = qasm.qubit_sizes.crbegin(); it != qasm.qubit_sizes.crend();
         ++it)
      regs.push_back(it->second);

  // Inner products
  if (target_states.empty() == false &&
      (show_final_inner_product || show_final_overlaps)) {
    // compute inner products
    cvector_t inprods;
    uint_t nstates = qreg.size();
    for (auto const &vec : target_states) {
      // check correct size
      if (vec.size() != nstates) {
        std::stringstream msg;
        msg << "error: target_state vector size \"" << vec.size()
            << "\" should be \"" << nstates << "\"";
        throw std::runtime_error(msg.str());
      }
      complex_t val = inner_product(vec, qreg);
      chop(val, epsilon);
      inprods.push_back(val);
    }

    // add output inner products
    if (show_final_inner_product)
      output_inprods.push_back(inprods);
    // Add output overlaps (needs renormalizing at output)
    if (show_final_overlaps)
      output_overlaps += get_probs(inprods);
  }

  // Density matrix (needs renormalizing at output)
  if (show_final_density) {
    const uint_t m = qreg.size();
    output_density.resize(m, m);
    output_density += outer_product(qreg, qreg);
  }

  // Final probabilities (needs renormalizing at output)
  if (show_final_probs) {
    output_probs.resize(qreg.size());
    for (uint_t j = 0; j < qreg.size(); j++) {
      double val = get_probs(qreg[j]);
      if (val > epsilon)
        output_probs[j] += val;
    }
  }

  // Final state ket or probabilites ket
  if (show_final_probs_ket || show_final_ket) {
    cket_t qregket = vec2ket(qreg, qudit_dim, epsilon, regs);
    // Final probabilities (needs renormalizing at output)
    if (show_final_probs)
      for (const auto &q : qregket)
        output_probs_ket[q.first] += get_probs(q.second);
    // Final state ket vectors
    if (show_final_ket)
      output_ket.push_back(qregket);
  }

  // Saved states
  if (qreg_saved.empty() == false) {

    // Ket form
    if (show_saved_ket || show_saved_probs_ket) {
      std::map<uint_t, cket_t> km;
      for (auto const &psi : qreg_saved)
        km[psi.first] = vec2ket(psi.second, qudit_dim, epsilon, regs);
      // saved kets
      if (show_saved_ket)
        saved_ket.push_back(km);
      // saved probabilities (ket form)
      if (show_saved_probs_ket)
        for (const auto &save : km) {
          rket_t tmp;
          for (const auto &vals : save.second)
            tmp[vals.first] = get_probs(vals.second);
          saved_probs_ket[save.first] += tmp;
        }
    }

    // add density matrix (needs renormalizing after all shots)
    if (show_saved_density) {
      for (auto const &psi : qreg_saved) {
        cmatrix_t &rho = saved_density[psi.first];
        if (rho.size() == 0)
          rho = outer_product(psi.second, psi.second);
        else
          rho = rho + outer_product(psi.second, psi.second);
      }
    }

    // add probs (needs renormalizing after all shots)
    if (show_saved_probs) {
      for (auto const &psi : qreg_saved) {
        auto &pr = saved_probs[psi.first];
        if (pr.empty())
          pr = get_probs(psi.second);
        else
          pr += get_probs(psi.second);
      }
    }
    // Inner products
    if (target_states.empty() == false &&
        (show_saved_inner_product || show_saved_overlaps)) {
      for (auto const &save : qreg_saved) {
        // compute inner products
        cvector_t inprods;
        uint_t nstates = qreg.size();
        for (auto const &vec : target_states) {
          // check correct size
          if (vec.size() != nstates) {
            std::stringstream msg;
            msg << "error: target_state vector size \"" << vec.size()
                << "\" should be \"" << nstates << "\"";
            throw std::runtime_error(msg.str());
          }
          complex_t val = inner_product(vec, save.second);
          chop(val, epsilon);
          inprods.push_back(val);
        }

        // add output inner products
        if (show_saved_inner_product)
          saved_inprods[save.first].push_back(inprods);
        // Add output overlaps (needs renormalizing at output)
        if (show_saved_overlaps)
          saved_overlaps[save.first] += get_probs(inprods);
      }
    }
  }
}

//------------------------------------------------------------------------------
double VectorEngine::get_probs(const complex_t &val) const {
  return std::real(std::conj(val) * val);
}

rvector_t VectorEngine::get_probs(const cvector_t &vec) const {
  rvector_t ret;
  for (const auto &elt : vec)
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
  const BaseEngine<cvector_t> &base_eng = eng;
  to_json(js, base_eng);

  // renormalization constant for average over shots
  double renorm = 1. / eng.total_shots;

  // add inner products
  if (eng.show_final_inner_product && eng.output_inprods.empty() == false) {
    auto tmp = eng.output_inprods;
    chop(tmp, eng.epsilon);
    js["inner_products"] = tmp;
  }

  if (eng.show_final_overlaps && eng.output_overlaps.empty() == false) {
    auto tmp = eng.output_overlaps * renorm;
    chop(tmp, eng.epsilon);
    js["overlaps"] = tmp;
  }

  // renormalize probs
  if (eng.show_final_probs && eng.output_probs.empty() == false) {
    rvector_t probs;
    for (const auto &ps : eng.output_probs) {
      double p = ps * renorm;
      probs.push_back((p < eng.epsilon) ? 0 : p);
    }
    js["probabilities"] = probs;
  }

  // renormalize probs ket
  if (eng.show_final_probs_ket && eng.output_probs_ket.empty() == false) {
    rket_t probs;
    for (const auto &p : eng.output_probs_ket)
      probs[p.first] = p.second * renorm;
    chop(probs, eng.epsilon);
    js["probabilities_ket"] = probs;
  }

  if (eng.show_final_ket && eng.output_ket.empty() == false) {
    js["quantum_states_ket"] = eng.output_ket;
  }

  // renormalize density
  if (eng.show_final_density && eng.output_density.size() > 0) {
    cmatrix_t rho = eng.output_density * renorm;
    chop(rho, eng.epsilon);
    js["density_matrix"] = rho;
  }

  // Saved kets
  if (eng.show_saved_ket && eng.saved_ket.empty() == false) {
    js["saved_quantum_states_ket"] = eng.saved_ket;
  }

  // Saved density
  if (eng.show_saved_density && eng.saved_density.empty() == false) {
    std::map<uint_t, cmatrix_t> saved_rhos;
    for (const auto &save : eng.saved_density) {
      auto rho = save.second * renorm;
      chop(rho, eng.epsilon);
      saved_rhos[save.first] = rho;
    }
    js["saved_density_matrix"] = saved_rhos;
  }
  // Saved probs
  if (eng.show_saved_probs && eng.saved_probs.empty() == false) {
    std::map<uint_t, rvector_t> ret;
    for (const auto &save : eng.saved_probs) {
      const auto &val = save.second;
      ret[save.first] = val * renorm;
      chop(ret[save.first], eng.epsilon);
    }
    js["saved_probabilities"] = ret;
  }
  // Saved probs ket
  if (eng.show_saved_probs_ket && eng.saved_probs_ket.empty() == false) {
    std::map<uint_t, rket_t> ret;
    for (const auto &save : eng.saved_probs_ket) {
      const auto &val = save.second;
      ret[save.first] = val * renorm;
      chop(ret[save.first], eng.epsilon);
    }
    js["saved_probabilities_ket"] = ret;
  }

  // Saved inner products
  if (eng.show_saved_inner_product && eng.saved_inprods.empty() == false) {
    auto tmp = eng.saved_inprods;
    for (auto &s : tmp)
      tmp[s.first] = chop(s.second, eng.epsilon);
    js["saved_inner_products"] = eng.saved_inprods;
  }
  // Saved overlaps
  if (eng.show_saved_overlaps && eng.saved_overlaps.empty() == false) {
    auto tmp = eng.saved_overlaps;
    for (auto &save : tmp) {
      save.second *= renorm;
      tmp[save.first] = chop(save.second, eng.epsilon);
    }
    js["saved_overlaps"] = tmp;
  }
}

inline void from_json(const json_t &js, VectorEngine &eng) {
  eng = VectorEngine();
  BaseEngine<cvector_t> &base_eng = eng;
  from_json(js, base_eng);
  // Get output options
  std::vector<std::string> opts;
  if (JSON::get_value(opts, "data", js)) {
    for (auto &o : opts) {
      to_lowercase(o);
      string_trim(o);

      if (o == "quantumstateket" || o == "quantumstatesket")
        eng.show_final_ket = true;
      else if (o == "densitymatrix")
        eng.show_final_density = true;
      else if (o == "probabilities" || o == "probs")
        eng.show_final_probs = true;
      else if (o == "probabilitiesket" || o == "probsket")
        eng.show_final_probs_ket = true;
      else if (o == "targetstatesinner")
        eng.show_final_inner_product = true;
      else if (o == "targetstatesprobs")
        eng.show_final_overlaps = true;

      else if (o == "savedquantumstateket" || o == "savedquantumstatesket")
        eng.show_saved_ket = true;
      else if (o == "saveddensitymatrix")
        eng.show_saved_density = true;
      else if (o == "savedprobabilities" || o == "savedprobs")
        eng.show_saved_probs = true;
      else if (o == "savedprobabilitiesket" || o == "savedprobsket")
        eng.show_saved_probs_ket = true;
      else if (o == "savedtargetstatesinner")
        eng.show_saved_inner_product = true;
      else if (o == "savedtargetstatesprobs")
        eng.show_saved_overlaps = true;
    }
  }
  // Get additional settings
  JSON::get_value(eng.epsilon, "chop", js);
  JSON::get_value(eng.qudit_dim, "qudit_dim", js);

  // renormalize state vector
  if (eng.initial_state_flag)
    renormalize(eng.initial_state);

  // parse target states from JSON
  bool renorm_target_states = true;
  JSON::get_value(renorm_target_states, "renorm_target_states", js);
  if (JSON::get_value(eng.target_states, "target_states", js) &&
      renorm_target_states)
    for (auto &v : eng.target_states)
      renormalize(v);
}

//------------------------------------------------------------------------------
} // end namespace QISKIT

#endif