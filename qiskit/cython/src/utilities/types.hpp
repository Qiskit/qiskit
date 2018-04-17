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
 * @file    types.h
 * @brief   miscellaneous functions
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _types_h_
#define _types_h_

#include <algorithm>
#include <bitset>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <stdint.h>

#include "qubit_vector.hpp"  // N-qubit vector class
#include "binary_vector.hpp" // Binary Vector class
#include "clifford.hpp"      // Clifford tableau class
#include "json.hpp"          // JSON Class library
#include "matrix.hpp"        // Matrix class library

/***************************************************************************/ /**
 *
 * Numeric Types for backends
 *
 ******************************************************************************/

// Numeric Types
using int_t = int64_t;
using uint_t = uint64_t;
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;
using cmatrix_t = matrix<complex_t>;
using rmatrix_t = matrix<double>;

// Timer
using myclock_t = std::chrono::system_clock;

// Custom Classes
using QV::QubitVector;

// Register Types
using svector_t = std::vector<std::string>;
using creg_t = std::vector<uint_t>;
using cket_t = std::map<std::string, complex_t>;
using rket_t = std::map<std::string, double>;

// Output types
using counts_t = std::map<std::string, uint_t>;
using count_pair_t = std::pair<std::string, uint_t>;

// JSON type
using json_t = nlohmann::json;

//------------------------------------------------------------------------------
// ostream overloads
//------------------------------------------------------------------------------

// STL containers
template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &out, const std::pair<T1, T2> &p);
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v);
template <typename T, size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &v);
template <typename T1, typename T2, typename T3>
std::ostream &operator<<(std::ostream &out, const std::map<T1, T2, T3> &m);
template <typename T1>
std::ostream &operator<<(std::ostream &out, const std::set<T1> &s);

// BinaryVector
inline std::ostream &operator<<(std::ostream &out, const BinaryVector &bv);
// QubitVector
inline std::ostream &operator<<(std::ostream &out, const QubitVector &qv);

/*******************************************************************************
 *
 * Gate_ID enumeration
 *
 * Enumeration class for indexing custom gates in simulation backends
 *
 ******************************************************************************/

enum class gate_t {

  // Core QASM 2.0 operations
  U,       // Arbitrary single qubit gate
  CX,      // Controlled-NOT gate
  Measure, // measure
  Reset,   // reset
  Barrier, // barrier

  // Single-qubit Clifford gates
  I,  // Pauli-I gate
  X,  // Pauli-X gate
  Y,  // Pauli-Y gate
  Z,  // Pauli-Z gate
  S,  // Phase gate aka Sqrt(Z) gate
  Sd, // Conjugate transpose of S
  H,  // Hadamard

  // Single-qubit non-clifford gates
  T,  // T-gate
  Td, // Conjugate transpose of T-gate
  U0, // idle gate in multiples of X90
  U1, // zero-X90 pulse waltz gate
  U2, // single-X90 pulse waltz gate
  U3, // two X90 pulse waltz gate

  // Custom Gates
  Wait, // Single qubit wait gate
  CZ,   // Controlled-phase gate
  RZZ,  // two-qubit zz-rotation

  // Simulator commands
  Snapshot, // save state of simulator for printing
  Noise, // gate to switch simulator noise on and off
  Save,  // save the current state of the qubit for later use
  Load   // load a previously saved qubit state into current qubit state
};

using gateset_t = std::map<std::string, gate_t>;


/***************************************************************************/ /**
  *
  * JSON Library Helper Functions
  *
  ******************************************************************************/

namespace JSON {

/**
 * Load a json_t from a file. If the file name is 'stdin' or '-' the json_t will
 * be
 * loaded from the standard input stream.
 * @param name: file name to load.
 * @returns: the loaded json.
 */
json_t load(std::string name);

/**
 * Check if a key exists in a json_t object.
 * @param key: key name.
 * @param js: the json_t to search for key.
 * @returns: true if the key exists, false otherwise.
 */
bool check_key(std::string key, const json_t &js);

/**
 * Check if all keys exists in a json_t object.
 * @param keys: vector of key names.
 * @param js: the json_t to search for keys.
 * @returns: true if all keys exists, false otherwise.
 */
bool check_keys(std::vector<std::string> keys, const json_t &js);

/**
 * Load a json_t object value into a variable if the key name exists.
 * @param var: variable to store key value.
 * @param key: key name.
 * @param js: the json_t to search for key.
 * @returns: true if the keys exists and val was set, false otherwise.
 */
template <typename T> bool get_value(T &var, std::string key, const json_t &js);

} // end namespace JSON

// Helper function to trim strings
void string_trim(std::string &str);

/***************************************************************************/ /**
  *
  * JSON Conversion for basic types
  *
  ******************************************************************************/

namespace std {

/**
 * Convert a complex number to a json list z -> [real(z), imag(z)].
 * @param js a json_t object to contain converted type.
 * @param z a complex number to convert.
 */
template <typename T> void to_json(json_t &js, const std::complex<T> &z);

/**
 * Convert a JSON value to a complex number z. If the json value is a float
 * it will be converted to a complex z = (val, 0.). If the json value is a
 * length two list it will be converted to a complex z = (val[0], val[1]).
 * @param js a json_t object to convert.
 * @param z a complex number to contain result.
 */
template <typename T> void from_json(const json_t &js, std::complex<T> &z);

/**
 * Convert a complex vector to a json list
 * v -> [ [real(v[0]), imag(v[0])], ...]
 * @param js a json_t object to contain converted type.
 * @param vec a complex vector to convert.
 */
void to_json(json_t &js, const cvector_t &vec);

/**
 * Convert a JSON list to a complex vector. The input JSON value may be:
 * - an object with complex pair values: {'00': [re, im], ... }
 * - an object with real pair values: {'00': n, ... }
 * - an list with complex values: [ [a0re, a0im], ...]
 * - an list with real values: [a0, a1, ....]
 * @param js a json_t object to convert.
 * @param vec a complex vector to contain result.
 */
void from_json(const json_t &js, cvector_t &vec);

/**
 * Convert a map with integer keys to a json. This converts the integer keys
 * to strings in the resulting json object.
 * @param js a json_t object to contain converted type.
 * @param map a map to convert.
 */
template <typename T> void to_json(json_t &js, const std::map<uint_t, T> &map);

} // end namespace std.

/**
 * Convert a matrix to JSON. This returns a list of list.
 * @param js a json_t object to contain converted type.
 * @param mat a matrix to convert.
 */
template <typename T> void to_json(json_t &js, const matrix<T> &mat);

/**
 * Convert a matrix to JSON. This returns a list of list.
 * @param js a json_t object to cconvert.
 * @param mat a matrix to to contain result.
 */
template <typename T> void from_json(const json_t &js, matrix<T> &mat);

/**
 * Convert a PauliOperator to JSON.
 */
void to_json(json_t &js, const PauliOperator &p);

/**
 * Parse a PauliOperator from JSON.
 */
void from_json(const json_t &js, PauliOperator &p);

/**
 * Convert a Clifford to JSON.
 */
void to_json(json_t &js, const Clifford &clif);

/**
 * Parse a Clifford from JSON.
 */
void from_json(const json_t &js, Clifford &clif);

namespace QV {
/**
 * Convert a QubitVector to JSON.
 */
void to_json(json_t &js, const QubitVector &qv);

/**
 * Parse a QubitVector from JSON.
 */
void from_json(const json_t &js, QubitVector &qv);
}

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Helper Functions
//------------------------------------------------------------------------------

json_t JSON::load(std::string name) {
  if (name == "") {
    json_t js;
    return js; // Return empty node if no config file
  }
  json_t js;
  if (name == "stdin" || name == "-") // Load from stdin
    // auto js = json::parse(read_stream(std::cin));
    std::cin >> js;
  else { // Load from file
    std::ifstream ifile;
    ifile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      ifile.open(name);
    } catch (std::exception &e) {
      throw std::runtime_error(std::string("no such file or directory"));
    }
    ifile >> js;
  }
  return js;
}

bool JSON::check_key(std::string key, const json_t &js) {
  // returns false if the value is 'null'
  if (js.find(key) != js.end() && !js[key].is_null())
    return true;
  else
    return false;
}

bool JSON::check_keys(std::vector<std::string> keys, const json_t &js) {
  bool pass = true;
  for (auto s : keys)
    pass &= check_key(s, js);
  return pass;
}

template <typename T>
bool JSON::get_value(T &var, std::string key, const json_t &js) {
  if (check_key(key, js)) {
    var = js[key].get<T>();
    return true;
  } else {
    return false;
  }
}

// Helper function to trim strings
void string_trim(std::string &str) {
  std::string tmp = "";
  for (auto c : str)
    if (c != ' ' && c != '_' && c != '-')
      tmp.push_back(c);
  str = tmp;
}
//------------------------------------------------------------------------------
// JSON Conversion
//------------------------------------------------------------------------------

template <typename T> void std::to_json(json_t &js, const std::complex<T> &z) {
  js = std::vector<T>{z.real(), z.imag()};
}

template <typename T>
void std::from_json(const json_t &js, std::complex<T> &z) {
  if (js.is_number())
    z = std::complex<T>{js.get<T>()};
  else if (js.is_array() && js.size() == 2) {
    z = std::complex<T>{js[0].get<T>(), js[1].get<T>()};
  } else {
    throw std::runtime_error(
        std::string("failed to parse json_t value as a complex number"));
  }
}

void std::to_json(json_t &js, const cvector_t &vec) {
  std::vector<rvector_t> out;
  for (auto &z : vec) {
    out.push_back(rvector_t{real(z), imag(z)});
  }
  js = out;
}

void std::from_json(const json_t &js, cvector_t &vec) {
  cvector_t ret;
  if (js.is_array()) {
    for (auto &elt : js)
      ret.push_back(elt);
    vec = ret;
  } else if (js.is_object()) {
    // deduce number of qubits from length of label string
    std::string refkey = js.begin().key();
    string_trim(refkey);
    uint_t nqubits = refkey.length();
    uint_t nstates = 1ull << nqubits;
    ret.resize(nstates);
    // import vector
    for (auto it = js.begin(); it != js.end(); ++it) {
      std::string key = it.key();
      string_trim(key);
      uint_t index = std::bitset<64>(key).to_ulong();
      std::complex<double> val = it.value().get<std::complex<double>>();
      ret[index] += val;
    }
    vec = ret;
  } else {
    throw std::runtime_error(
        std::string("failed to parse json_t value as a complex vector"));
  }
}

// Matrices
template <typename T> void to_json(json_t &js, const matrix<T> &mat) {
  json_t ret;
  size_t rows = mat.GetRows();
  size_t cols = mat.GetColumns();
  for (uint_t r = 0; r < rows; r++) {
    std::vector<T> mrow;
    for (uint_t c = 0; c < cols; c++)
      mrow.push_back(mat(r, c));
    ret.push_back(mrow);
  }
  js = ret;
}

template <typename T> void from_json(const json_t &js, matrix<T> &mat) {
  // Check it is a non empty array
  bool is_matrix = js.is_array() && !js.empty();
  // Check all entries of array are same size
  size_t cols = js[0].size();
  size_t rows = js.size();
  for (auto &row : js)
    is_matrix &= (row.is_array() && row.size() == cols);

  // Convert
  if (is_matrix) {
    matrix<T> ret(rows, cols);
    for (uint_t r = 0; r < rows; r++)
      for (uint_t c = 0; c < cols; c++)
        ret(r, c) = js[r][c].get<T>();
    mat = ret;
  } else {
    throw std::runtime_error(
        std::string("failed to parse json_t value as a matrix"));
  }
}

// Int-key maps
template <typename T>
void std::to_json(json_t &js, const std::map<uint_t, T> &map) {
  js = json_t();
  for (const auto &p : map) {
    std::string key = std::to_string(p.first);
    js[key] = p.second;
  }
}

// PauliOperator
void to_json(json_t &js, const PauliOperator &p) {
  json_t tmp;
  tmp["X"] = p.X.getData();
  tmp["Z"] = p.Z.getData();
  tmp["phase"] = static_cast<uint_t>(p.phase);
  js = tmp;
}

void from_json(const json_t &js, PauliOperator &p) {
  if (JSON::check_keys({"pahse", "X", "Z"}, js)) {
    PauliOperator tmp;
    std::vector<uint_t> x = js["X"], z = js["Z"];
    tmp.phase = js["phase"];
    tmp.X = BinaryVector(x);
    tmp.Z = BinaryVector(z);
    p = tmp;
  } else {
    throw std::runtime_error(
        std::string("failed to parse json_t value as a PauliOperator"));
  }
}

// Clifford
void to_json(json_t &js, const Clifford &clif) {
  // first n rows are destabilizers; last n rows are stabilizers
  // we don't print the auxillary row
  const auto table = clif.get_table();
  uint_t n = (table.size() - 1) / 2;
  assert(2 * n + 1 == table.size());
  for (uint_t j = 0; j < n; j++) {
    js["destabilizers"].push_back(table[j]);
  }
  for (uint_t j = n; j < 2 * n; j++) {
    js["stabilizers"].push_back(table[j]);
  }
}

void from_json(const json_t &js, Clifford &clif) {
  if (js.is_object() &&
      JSON::check_keys({"stabilizers", "destabilizers"}, js)) {
    // Stored as kkeyed object
    const json_t &stab = js["stabilizers"];
    const json_t &destab = js["destabilizers"];
    size_t nq = stab.size();
    clif = Clifford(nq);
    for (size_t j = 0; j < nq; j++)
      clif[j] = destab[j].get<PauliOperator>();
    for (size_t j = 0; j < nq; j++)
      clif[nq + j] = stab[j].get<PauliOperator>();
  } else if (js.is_array() && js.size() % 2 == 0) {
    // Stored as 2 * nq array
    auto l = js.size();
    uint_t nq = l / 2; // size 2 * nq array
    clif = Clifford(nq);
    for (size_t j = 0; j < l; j++) {
      PauliOperator p = js[j];
      clif[j] = p;
    }
  } else {
    throw std::runtime_error(
        std::string("failed to parse json_t value as a Clifford"));
  }
}

void QV::to_json(json_t &js, const QubitVector &qv) {
  to_json(js, qv.vector());
}

void QV::from_json(const json_t &js, QubitVector &qv) {
  cvector_t tmp;
  from_json(js, tmp);
  qv = tmp;
}

//------------------------------------------------------------------------------
// ostream overloads
//------------------------------------------------------------------------------

// ostream overload for pairs
template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &out, const std::pair<T1, T2> &p) {
  out << "(" << p.first << ", " << p.second << ")";
  return out;
}

// ostream overload for vectors
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  out << "[";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "]";
  return out;
}

// ostream overload for arrays
template <typename T, size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<T, N> &v) {
  out << "[";
  for (size_t i = 0; i < N; ++i) {
    out << v[i];
    if (i != N - 1)
      out << ", ";
  }
  out << "]";
  return out;
}

// ostream overload for maps
template <typename T1, typename T2, typename T3>
std::ostream &operator<<(std::ostream &out, const std::map<T1, T2, T3> &m) {
  out << "{";
  size_t pos = 0, last = m.size() - 1;
  for (auto const &p : m) {
    out << p.first << ":" << p.second;
    if (pos != last)
      out << ", ";
    pos++;
  }
  out << "}";
  return out;
}

// ostream overload for sets
template <typename T1>
std::ostream &operator<<(std::ostream &out, const std::set<T1> &s) {
  out << "{";
  size_t pos = 0, last = s.size() - 1;
  for (auto const &elt : s) {
    out << elt;
    if (pos != last)
      out << ", ";
    pos++;
  }
  out << "}";
  return out;
}

// ostream overload for BinaryVector
inline std::ostream &operator<<(std::ostream &out, const BinaryVector &bv) {
  out << bv.getData();
  return out;
}

// ostream overload for QubitVector
inline std::ostream &operator<<(std::ostream &out, const QubitVector &qv) {
  out << qv.vector();
  return out;
}

//------------------------------------------------------------------------------
#endif
