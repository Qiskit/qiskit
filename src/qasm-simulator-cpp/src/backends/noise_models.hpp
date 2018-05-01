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
 * @file    noise_models.hpp
 * @brief   Noise Models for Simulator Backends
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _NoiseModels_h_
#define _NoiseModels_h_

#include <complex>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include "misc.hpp"
#include "types.hpp"

namespace QISKIT {

/*******************************************************************************
 *
 * Relaxation Class
 *
 ******************************************************************************/

class Relaxation {
public:
  /**
   * Relaxation data members
   */
  double rate = 0.;                            // relxation rate
  std::discrete_distribution<> populations{1}; // equilibrium populations

  /**
   * Computes the relaxation probability for a length of time assuming
   * exponential relaxation: p(t) = 1-exp(-t*rate).
   * @param t: the length of time
   * @returns: relaxation probability p(t)
   */
  double p(double t) const;

  /**
   * Default constructor: no relaxation error
   */
  Relaxation(){};

  /**
   * Constructor: sets the relaxation rate and equilibrium population that
   * system will relax towards.
   * @param rate1: the relaxation rate
   * @param pops: equilibrium populations as a discrete_distribution
   */
  Relaxation(double rate1, const std::discrete_distribution<> &pops);

  /**
   * Constructor: sets the relaxation rate and equilibrium population that
   * system will relax towards. If the input populations are not normalized
   * they will be rescaled when converted to a discrete_distribution.
   * @param rate1: the relaxation rate
   * @param pops: vector of equilibrium populations
   */
  Relaxation(double rate1, const rvector_t &pops);

  /**
   * Verify that the set noise parameters are valid for the simulation by
   * checking that the relaxation rate is non-negative, and that the vector of
   * probabilities for the equilibrium distribution is not greater than the
   * subsystem dimension dim.
   * @param dim: dimension of each subsystem (typically 2 for a qubit)
   * @returns: true if noise parameters are valid
   */
  bool verify(uint_t dim) const;
};

/*******************************************************************************
 *
 * Reset Error Class
 *
 * This class specifies errors in the QASM reset operation. The reset error sets
 * a system to the wrong state with some probability following a reset.
 *
 ******************************************************************************/

class ResetError {
public:
  /***
   * ResetError data members
   ***/
  bool ideal = true;                 // flag if reset error is set
  std::discrete_distribution<> p{1}; // the reset error probabilities

  /***
   * Default constructor: no reset error
   ***/
  ResetError(){};

  /***
   * Constructor: sets the probabilities of resetting to each energy level using
   * a discrete_distribution input.
   * @param probs: reset probabilities as a discrete_distribution
   ***/
  ResetError(const std::discrete_distribution<> &probs);

  /***
   * Constructor: sets the probabilities of resetting to each energy level using
   * a vector of probabilities as input. If the input probabilities are not
   * normalized they will be rescaled when converted to a discrete_distribution.
   * @param probs: reset probabilities as a vector
   ***/
  ResetError(const rvector_t &probs);

  /***
   * Verify that the set noise parameters are valid for the simulation by
   * checking that the length of the vector of reset probabilities is not
   *greater
   * than the subsystem dimension dim.
   * @param dim: dimension of each subsystem (typically 2 for a qubit)
   * @returns: true if noise parameters are valid
   ***/
  bool verify(uint_t dim) const;
};

/*******************************************************************************
 *
 * ReadoutError Class
 *
 * This class specifies errors in recording the value of the classical bit after
 * performing a QASM measure operation. It is an assignment error where the
 * probability of recording a measurement outcome as k, given that m was
 * actually obtained is given by P(k|m).
 *
 ******************************************************************************/

class ReadoutError {
public:
  /***
   * ResetError data members
   ***/
  bool ideal = true;                           // flag if reset error is set
  std::vector<std::discrete_distribution<>> p; // the assignment error probs

  /***
   * Default constructor: no readout error
   ***/
  ReadoutError(){};

  /***
   * Constructor: sets the readout assignment probablities from a vector of
   * discrete_distributions for each measurement outcome.
   * @param probs: vector of discrete_distributions
   ***/
  ReadoutError(const std::vector<std::discrete_distribution<>> &probs);

  /***
   * Constructor: sets the readout assignment probablities to be of the form
   *  P(0|0) = 1-p1, P(1|0) = p1 for a 0 measurement outcome
   *  P(0|1) = p1, P(1|1) = 1-p1 for a 1 measurement outcome
   * @param p1: assignement error probability
   ***/
  explicit ReadoutError(double p1);

  /***
   * Constructor: sets the readout assignment probablities from a vector of
   * probabilty vectors for each measurement outcome. If any probability
   * vector is not correctly normalized it will be rescaled when converted to
   * a discrete_distribution.
   * @param probs: vector of probability vectors
   ***/
  ReadoutError(const std::vector<rvector_t> &probs);

  /***
   * Verify that the set noise parameters are valid for the simulation by
   * checking that the vector of assignment probabilities is the correct length
   *for
   * the subsystem dimension dim.
   * @param dim: dimension of each subsystem (typically 2 for a qubit)
   * @returns: true if noise parameters are valid
   ***/
  bool verify(uint_t dim) const;
};

/*******************************************************************************
 *
 * PauliChannel Class
 *
 ******************************************************************************/

class PauliChannel {
public:
  uint_t n = 0;
  bool ideal = true;
  std::discrete_distribution<> p;

  // Constructors
  PauliChannel(){};
  explicit PauliChannel(uint_t nq) : n(nq){};
  PauliChannel(uint_t nq, rvector_t p_pauli);

  // Get vector of error probabilities eg {pX, pY, pZ}
  rvector_t p_errors() const;
  void set_p(rvector_t p_pauli);
};

/*******************************************************************************
 *
 * GateError Class
 *
 ******************************************************************************/

class GateError {
public:
  /***
   * GateError data members
   ***/
  std::string label; // label for gate error
  bool ideal = true; // flag if any gate error is set

  /***
   * Incoherent Error Channel parameters
   ***/
  PauliChannel pauli;
  double gate_time = 0.;

  /***
   * Unitary Coherent Error Channel parameters
   ***/
  bool coherent_error = false; // flag if coherent gate error is set
  cmatrix_t Uerr;              // Coherent error to be applied after gate

  /***
   * Default constructor: no gate error
   ***/
  GateError(){};

  /***
   * Verify that the set noise parameters are valid for the simulation by
   * checking that all probabilities are non-negative, the length of pauli
   *error
   * probabiltity vectors are the correct for the size of the gate, and the
   * coherent error matrices are the correct dimension for the subsystem size
   * @param dim: dimension of each subsystem (typically 2 for a qubit)
   * @returns: true if noise parameters are valid
   ***/
  bool verify(uint_t dim) const;
};

/*******************************************************************************
 *
 * QubitNoise Class
 *
 ******************************************************************************/

class QubitNoise {
public:
  /***
   * Noise members
   ***/
  bool ideal = true;    // flag if any error is set
  ResetError reset;     // noise params used by QASM reset
  ReadoutError readout; // noise params used by QASM measure
  Relaxation relax;     // relaxation parameters used by other errors
  std::map<std::string, GateError> gate; // noise params for each gate type
  const static std::vector<std::string> gate_names;

  /***
   * Default constructor: no gate error
   ***/
  QubitNoise(){};

  /***
   * Verify that each of the noise class members has valid parameters by
   *calling
   * the corresponding verify method of each member.
   * @param dim: dimension of each subsystem (typically 2 for a qubit)
   * @returns: true if noise parameters are valid
   ***/
  bool verify(uint_t dim = 2);
};

/*******************************************************************************
 *
 * Relaxation Class Methods
 *
 ******************************************************************************/

double Relaxation::p(double t) const {
  return (rate > 0. && t > 0.) ? 1. - std::exp(-t * rate) : 0.;
}

Relaxation::Relaxation(double rate1, const std::discrete_distribution<> &pops)
    : rate(rate1), populations(pops) {
#ifdef DEBUG
  std::cout << "DEBUG Constructor: Relaxation" << std::endl;
  std::cout << "DEBUG relaxation_rate = " << rate << std::endl;
  std::cout << "DEBUG thermal_populations = " << populations.probabilities()
            << std::endl;
#endif
}

Relaxation::Relaxation(double rate1, const rvector_t &pops)
    : Relaxation(rate1,
                 std::discrete_distribution<>(pops.begin(), pops.end())) {}

bool Relaxation::verify(uint_t dim) const {
  // Check Relaxation
  if (populations.probabilities().size() > dim) {
    std::cerr << "error: thermal_populations vector is too long" << std::endl;
    return false;
  } else if (rate < 0) {
    std::cerr << "error: relaxation_rate is negative" << std::endl;
    return false;
  } else
    return true;
}

inline void to_json(json_t &js, const Relaxation &error) {
  if (error.rate > 0.) {
    js["relaxation_rate"] = error.rate;
    js["thermal_populations"] = error.populations.probabilities();
  }
}

inline void from_json(const json_t &noise, Relaxation &error) {
  if (JSON::check_key("relaxation_rate", noise)) {
    double rate = noise["relaxation_rate"];
    rvector_t pops{1.};
    if (JSON::check_key("thermal_populations", noise)) {
      const json_t &node = noise["thermal_populations"];
      if (node.is_number())
        pops[0] = node.get<double>();
      else
        pops = node.get<rvector_t>();
    }
    error = Relaxation(rate, pops);
  } else
    error = Relaxation();
}

/*******************************************************************************
 *
 * ResetError Methods
 *
 ******************************************************************************/

ResetError::ResetError(const std::discrete_distribution<> &probs) : p(probs) {
  ideal = !(p.probabilities()[0.] < 1.);
#ifdef DEBUG
  std::cout << "DEBUG Constructor: ResetError" << std::endl;
  std::cout << "DEBUG p = " << p.probabilities() << std::endl;
#endif
}

ResetError::ResetError(const rvector_t &probs)
    : ResetError(std::discrete_distribution<>(probs.begin(), probs.end())) {}

bool ResetError::verify(uint_t dim) const {
  if (p.probabilities().size() > dim) {
    std::cerr << "error: reset.p error vector is too long" << std::endl;
    return false;
  } else
    return true;
}

inline void to_json(json_t &js, const ResetError &error) {
  if (!error.ideal && error.p.probabilities()[0] < 1.) {
    js["reset_error"] = error.p.probabilities();
  }
}

inline void from_json(const json_t &noise, ResetError &error) {
  if (JSON::check_key("reset_error", noise)) {
    // Scalar value is p0, 1-p0 reset error
    if (noise["reset_error"].is_number()) {
      double p1 = noise["reset_error"].get<double>();
      error = (p1 > 0.) ? ResetError(std::discrete_distribution<>({1 - p1, p1}))
                        : ResetError();
    }
    // node is a vector of reset probs
    else if (noise["reset_error"].is_array()) {
      error = ResetError(noise["reset_error"].get<rvector_t>());
    } else {
      throw std::runtime_error(std::string("p_reset error invalid input"));
    }
  } else
    error = ResetError();
}

/*******************************************************************************
 *
 * ReadoutError Methods
 *
 ******************************************************************************/

ReadoutError::ReadoutError(
    const std::vector<std::discrete_distribution<>> &probs)
    : ideal(false), p(probs) {
#ifdef DEBUG
  std::cout << "DEBUG Constructor: ReadoutError" << std::endl;
  for (const auto &v : p)
    std::cout << "DEBUG p = " << v.probabilities() << std::endl;
#endif
}

ReadoutError::ReadoutError(double p1) {
  if (p1 > 0.) {
    ideal = false;
    p.push_back(std::discrete_distribution<>{1. - p1, p1});
    p.push_back(std::discrete_distribution<>{p1, 1. - p1});
  }
#ifdef DEBUG
  std::cout << "DEBUG Constructor: ReadoutError" << std::endl;
  for (const auto &v : p)
    std::cout << "DEBUG p = " << v.probabilities() << std::endl;
#endif
}

ReadoutError::ReadoutError(const std::vector<rvector_t> &probs) {
  ideal = false;
  for (auto const &v : probs)
    p.push_back(std::discrete_distribution<>(v.begin(), v.end()));

#ifdef DEBUG
  std::cout << "DEBUG Constructor: ReadoutError" << std::endl;
  for (const auto &v : p)
    std::cout << "DEBUG p = " << v.probabilities() << std::endl;
#endif
}

bool ReadoutError::verify(uint_t dim) const {
  if (p.size() > dim) {
    std::cerr << "error: readout_error.p error vector is too long" << std::endl;
    return false;
  } else {
    for (const auto &d : p) {
      if (d.probabilities().size() > dim) {
        std::cerr << "error: readout_error.p error vector is too long"
                  << std::endl;
        return false;
      }
    }
    return true;
  }
}

inline void to_json(json_t &js, const ReadoutError &error) {
  if (!error.ideal) {
    std::vector<rvector_t> probs;
    for (auto const &dist : error.p)
      probs.push_back(dist.probabilities());
    js["readout_error"] = probs;
  }
}

inline void from_json(const json_t &noise, ReadoutError &error) {
  if (JSON::check_key("readout_error", noise)) {
    const json_t &node = noise["readout_error"];
    // Scalar value is p0, 1-p0 reset error
    if (node.is_number()) {
      error = ReadoutError(node.get<double>());
    } else if (node.is_array()) {
      // check length of array
      if (node[0].is_number() && node.size() == 2) {
        // 2-outcome only assignment errors
        double p0 = node[0];
        double p1 = node[1];
        if (p1 > 0. || p0 > 0.) {
          std::vector<std::discrete_distribution<>> ps;
          ps.push_back(std::discrete_distribution<>({1 - p0, p0}));
          ps.push_back(std::discrete_distribution<>({p1, 1 - p1}));
          error = ReadoutError(ps);
        } else {
          error = ReadoutError();
        }
      } else {
        std::vector<rvector_t> ps = node;
        error = ReadoutError(ps);
      }
    } else {
      throw std::runtime_error(std::string("p_meas_error vector invalid"));
    }
  } else
    error = ReadoutError();
}

/*******************************************************************************
 *
 * PauliChannel Methods
 *
 ******************************************************************************/

PauliChannel::PauliChannel(uint_t nq, rvector_t p_pauli) : n(nq) {
  set_p(p_pauli);
}

void PauliChannel::set_p(rvector_t p_pauli) {
  // Get pI error prob
  uint_t N = 1ULL << (2 * n);
  double tot = 0;
  for (auto &pj : p_pauli)
    tot += pj;
  p_pauli.insert(p_pauli.begin(), std::max(0., 1. - tot));
  // Check vector
  if (p_pauli.size() > N || tot > 1. || tot < 0.)
    throw std::runtime_error("invalid Pauli vector");
  if (p_pauli[0] < 1.) {
    p = std::discrete_distribution<>(p_pauli.begin(), p_pauli.end());
    ideal = false;
  }
}

rvector_t PauliChannel::p_errors() const {
  const auto d = (1ULL << n);
  rvector_t ret = p.probabilities();
  ret.erase(ret.begin()); // remove pI element
  ret.resize(d * d - 1);  // resize
  return ret;
}

/*******************************************************************************
 *
 * GateError Methods
 *
 ******************************************************************************/

bool GateError::verify(uint_t dim) const {
  if (pauli.p.probabilities().size() > dim * dim) {
    std::cerr << "error: pauli error vector is wrong length." << std::endl;
    return false;
  } else if (gate_time < 0.) {
    std::cerr << "error: gate_time must be non-negative" << std::endl;
    return false;
  } else
    return true;
}

inline void to_json(json_t &js, const GateError &error) {
  if (!error.ideal) {
    json_t node;
    if (error.gate_time > 0.)
      node["gate_time"] = error.gate_time;
    if (error.pauli.ideal == false)
      node["p_pauli"] = error.pauli.p_errors();
    if (error.coherent_error && error.Uerr.size() > 0)
      node["U_error"] = error.Uerr;
    js[error.label] = node;
  }
}

inline GateError load_gate_error(std::string key, uint_t nq, const json_t &js) {

  GateError error = GateError();
  error.label = key;
  json_t noise;
  if (JSON::check_key(key, js))
    noise = js[key];
  else
    noise = js;

  // Load Gate Time
  JSON::get_value(error.gate_time, "gate_time", noise);
  if (error.gate_time > 0.)
    error.ideal = false;

  // Load Coherent Error
  if (JSON::check_key("U_error", noise)) {
    error.ideal = false;
    error.coherent_error = true;
    error.Uerr = noise["U_error"];
  }

  // Load Pauli Error
  double p_depol = 0.;
  rvector_t p_pauli;
  uint_t N = 1ULL << (2 * nq);
  JSON::get_value(p_pauli, "p_pauli", noise);
  if (JSON::get_value(p_depol, "p_depol", noise) && p_depol > 0.) {
    p_pauli.resize(N - 1);
    for (auto &p : p_pauli)
      p = p + (p_depol / double(N)) - (p_depol * p);
  }
  error.pauli = PauliChannel(nq, p_pauli);
  error.ideal &= error.pauli.ideal;
  return error;
}

/*******************************************************************************
 *
 * QubitNoise Methods
 *
 ******************************************************************************/

bool QubitNoise::verify(uint_t dim) {
  bool pass = reset.verify(dim) && readout.verify(dim) && relax.verify(dim);

  for (const auto &g : gate) {
    uint_t dim2 = (g.first == "CX" || g.first == "CZ") ? dim * dim : dim;
    pass = pass && g.second.verify(dim2);
  }
  return pass;
}

const std::vector<std::string>
    QubitNoise::gate_names({"X90", "CX", "CZ", "id", "U", "measure", "reset"});

inline void to_json(json_t &js, const QubitNoise &noise) {
  json_t node;
  to_json(node, noise.relax);
  to_json(node, noise.reset);
  to_json(node, noise.readout);
  for (const auto &g : noise.gate) {
    to_json(node, g.second);
  }
  js = node;
}

inline void from_json(const json_t &js, QubitNoise &noise) {
  if (JSON::check_key("noise_params", js)) {
    from_json(js["noise_params"], noise);
  } else {
    noise = QubitNoise();
    from_json(js, noise.reset);
    noise.ideal &= noise.reset.ideal;
    noise.readout = js;
    noise.ideal &= noise.readout.ideal;
    noise.relax = js;
    noise.ideal &= !(noise.relax.rate > 0.);
    for (const auto &n : QubitNoise::gate_names) {
      if (JSON::check_key(n, js)) {
        GateError g;
        if (n == "CX" || n == "CZ")
          g = load_gate_error(n, 2, js);
        else
          g = load_gate_error(n, 1, js);

        // Check coherent error
        if (g.coherent_error) {
          cmatrix_t check = MOs::Dagger(g.Uerr) * g.Uerr;
          double threshold = 1e-10;
          double delta = 0.;
          for (size_t i=0; i < check.GetRows(); i++)
            for (size_t j=0; j < check.GetColumns(); j++) {
              complex_t val = (i==j) ? 1. : 0.;
              delta += std::real(std::abs(check(i, j) - val));
            }
          if (delta > threshold) {
            throw std::runtime_error(std::string(g.label + " U_error is not unitary"));
          }
        }

        // Add gate to noise model
        noise.gate.insert(std::make_pair(n, g));
        noise.ideal &= g.ideal;
      }
    }
  }
}

} // end namespace QISKIT
#endif