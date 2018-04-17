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
 * @file rng_engine.hpp
 * @brief RngEngine use by the BaseBackend simulator class
 * @author Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _RngEngine_hpp_
#define _RngEngine_hpp_

#include <random>

#include "types.hpp"

/***************************************************************************/ /**
  *
  * RngEngine Class
  *
  * Objects of this class are used to generate random numbers for backends.
  *These
  * are used to decide outcomes of measurements and resets, and for implementing
  * noise.
  *
  ******************************************************************************/

class RngEngine {
public:
  /**
   * Generate a uniformly distributed pseudo random real in the half-open
   * interval [a,b)
   * @param a closed lower bound on interval
   * @param b open upper bound on interval
   * @return the generated double
   */
  double rand(double a, double b);

  /**
   * Generate a uniformly distributed pseudo random real in the half-open
   * interval [0,b)
   * @param b open upper bound on interval
   * @return the generated double
   */
  inline double rand(double b) { return rand(0., b); };

  /**
   * Generate a uniformly distributed pseudo random real in the half-open
   * interval [0,1)
   * @return the generated double
   */
  inline double rand() { return rand(0., 1.); };

  /**
   * Generate a uniformly distributed pseudo random integer in the closed
   * interval [a,b]
   * @param a lower bound on interval
   * @param b upper bound on interval
   * @return the generated integer
   */
  int_t rand_int(int_t a, int_t b);

  /**
   * Generate a pseudo random integer from an input discrete distribution
   * @param probs the discrete distribution to sample from
   * @return the generated integer
   */
  int_t rand_int(std::discrete_distribution<> probs);

  /**
   * Generate a pseudo random integer from a a discrete distribution
   * constructed from an input vector of probabilities for [0,..,n-1]
   * where n is the lenght of the vector. If this vector is not normalized
   * it will be scaled when it is converted to a discrete_distribution
   * @param probs the vector of probabilities
   * @return the generated integer
   */
  int_t rand_int(std::vector<double> probs);

  /**
   * Default constructor initialize RNG engine with a random seed
   */
  RngEngine() {
    std::random_device rd;
    rng.seed(rd());
  };

  /**
   * Seeded constructor initialize RNG engine with a fixed seed
   * @param seed integer to use as seed for mt19937 engine
   */
  explicit RngEngine(uint_t seed) { rng.seed(seed); };

private:
  std::mt19937 rng; // Mersenne twister rng engine
};

/*******************************************************************************
 *
 * RngEngine Methods
 *
 ******************************************************************************/

double RngEngine::rand(double a, double b) {
  double p = std::uniform_real_distribution<double>(a, b)(rng);
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG: rand(" << a << "," << b << ") = " << p;
  std::clog << ss.str() << std::endl;
#endif
  return p;
}

// randomly distributed integers in [a,b]
int_t RngEngine::rand_int(int_t a, int_t b) {
  uint_t n = std::uniform_int_distribution<>(a, b)(rng);
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG: rand_int( " << a << "," << b << ") = " << n;
  std::clog << ss.str() << std::endl;
#endif
  return n;
}

// randomly distributed integers from discrete distribution
int_t RngEngine::rand_int(std::discrete_distribution<> probs) {
  uint_t n = probs(rng);
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG: rand_int(" << probs.probabilities() << ") = " << n;
  std::clog << ss.str() << std::endl;
#endif
  return n;
}

// randomly distributed integers from vector
int_t RngEngine::rand_int(std::vector<double> probs) {
  uint_t n = std::discrete_distribution<>(probs.begin(), probs.end())(rng);
#ifdef DEBUG
  std::stringstream ss;
  ss << "DEBUG: rand_int(" << probs << ") = " << n;
  std::clog << ss.str() << std::endl;
#endif
  return n;
}

//------------------------------------------------------------------------------
#endif
