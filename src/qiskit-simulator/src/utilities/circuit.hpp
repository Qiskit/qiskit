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
 * @file    circuit.hpp
 * @brief   Compiled circuit data structure for SimBackends
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _circuit_h_
#define _circuit_h_

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "misc.hpp"
#include "types.hpp"

/***************************************************************************/ /**
  *
  * Numeric Types for backends
  *
  ******************************************************************************/

// using json = nlohmann::json;

namespace QISKIT {

// typedef std::map<std::string, gate_t> GateSet;
typedef std::pair<std::string, uint_t> regType;
typedef std::vector<regType> reglist;

/*******************************************************************************
 *
 * operation_if struct
 *
 ******************************************************************************/

struct operation_if {
public:
  std::string type; // boolean type
  creg_t mask;
  creg_t val;
};

/*******************************************************************************
 *
 * operation struct
 *
 ******************************************************************************/

struct operation {
public:
  gate_t id;
  std::string name;
  std::vector<double> params;
  creg_t qubits;
  creg_t clbits;
  bool if_op = false;
  operation_if cond;
};

/*******************************************************************************
 *
 * Circuit class
 *
 ******************************************************************************/

class Circuit {
public:
  uint_t nqubits; // number of qubits
  uint_t nclbits; // number of classical bits
  reglist qubit_labels;
  reglist clbit_labels;
  std::vector<operation> operations;

  // new
  reglist qubit_sizes;
  std::string name = "";
  int_t rng_seed = -1;   // backend rng seed
  uint_t shots = 1;      // number of simulation shots
  bool opt_meas = false; // true if all measurements at end
  json_t config;         // local config

  /**
   * Default Constructor
   */
  Circuit(){};

  /**
   * Partial Constructor
   */
  inline Circuit(const json_t &circuit) {
    const gateset_t gs;
    const json_t qobjconf;
    parse(circuit, qobjconf, gs);
  };

  /**
   * Full Constructor
   */
  inline Circuit(const json_t &circuit, const json_t &qobjconf,
                 const gateset_t &gs) {
    parse(circuit, qobjconf, gs);
  };

  /**
   *  Parse a json qobj circuit
   */
  void parse(const json_t &circuit, const json_t &qobjconf,
             const gateset_t &gs);

private:
  /**
   *  Parse a json qobj circuit operation
   */
  operation parse_op(const json_t &js, const gateset_t &gs);

  /**
   *  Parse a json qobj circuit operation conditional
   */
  operation_if parse_conditional(const json_t &jcond);

  /**
   *  Parse the qubit and clbit registers from json qobj
   */
  reglist parse_reglist(const json_t &js);

  /**
   *  Set the Gate type for a given gate string
   */
  bool set_gateid(operation &op, std::string name, const gateset_t &gs);

  /**
   *  Return true if tail of circuit operaitons are all measurements
   */
  bool check_opt_meas();
};

/*******************************************************************************
 *
 * QISKIT methods
 *
 ******************************************************************************/

void Circuit::parse(const json_t &circuit, const json_t &qobjconf,
                    const gateset_t &gs) {
#ifdef DEBUG
  std::clog << "DEBUG (json): parsing circuit object" << std::endl;
#endif
  // Parse header
  const json_t &header = circuit["compiled_circuit"]["header"];

  JSON::get_value(nqubits, "number_of_qubits", header);
  JSON::get_value(nclbits, "number_of_clbits", header);
  qubit_labels = parse_reglist(header.at("qubit_labels"));
  clbit_labels = parse_reglist(header.at("clbit_labels"));

  // Store qubit registers like clbit ([[q1, sz1], [q2, sz2]])
  for (auto reg : qubit_labels) {
    if (qubit_sizes.empty() == false && reg.first == qubit_sizes.back().first) {
      qubit_sizes.back().second++;
    } else
      qubit_sizes.push_back({reg.first, 1});
  }

  // Check qubit labels
  if (nqubits != qubit_labels.size()) {
    throw std::runtime_error(
        std::string("number_qubits does not match qubit_labels"));
  }
#ifdef DEBUG
  std::clog << "DEBUG (json): qubit_labels = " << qubit_labels << std::endl;
#endif

  // Check clbit labels
  uint_t count = 0;
  for (const auto &reg : clbit_labels)
    count += reg.second;
  if (count != nclbits) {
    throw std::runtime_error(
        std::string("number_clbits does not match clbit_labels"));
  }
  if (nqubits != qubit_labels.size()) {
    throw std::runtime_error(
        std::string("number_qubits does not match qubit_labels"));
  }
#ifdef DEBUG
  std::clog << "DEBUG (json): clbit_labels = " << clbit_labels << std::endl;
  std::clog << "DEBUG (json): parsing operations" << std::endl;
#endif

  // parse operations
  const json_t &ops = circuit["compiled_circuit"]["operations"];
  for (auto it = ops.begin(); it != ops.end(); ++it)
    operations.push_back(parse_op(*it, gs));
  opt_meas = check_opt_meas(); // check measurement optimization

  // Load optional values
  JSON::get_value(name, "name", circuit); // look for circuit name

  // Parse Config
  config = qobjconf; // copy qobj level config
  if (JSON::check_key("config", circuit)) {
    for (auto it = circuit["config"].cbegin(); it != circuit["config"].cend();
         ++it) {
      config[it.key()] = it.value(); // overwrite circuit level config values
    }
  }

  // load config
  JSON::get_value(shots, "shots", config);
  JSON::get_value(rng_seed, "seed", config);
}

//------------------------------------------------------------------------------
bool Circuit::set_gateid(operation &op, std::string label,
                         const gateset_t &gs) {
  auto pos = gs.find(label);
  if (pos != gs.end()) {
    op.id = pos->second;
    op.name = label;
    return true;
  } else
    return false;
}

//------------------------------------------------------------------------------
operation Circuit::parse_op(const json_t &node, const gateset_t &gs) {

  operation op;
  std::string label;

  // Check operation is in the gateset
  if (!(node.is_object() && JSON::get_value(label, "name", node) &&
        set_gateid(op, label, gs))) {
    throw std::runtime_error(
        std::string("invalid operation \'" + label + "\'."));
  }

  // load op parameters
  JSON::get_value(op.params, "params", node);
  JSON::get_value(op.qubits, "qubits", node);
  JSON::get_value(op.clbits, "clbits", node);

  // Check op
  for (auto q : op.qubits)
    if (q >= nqubits) {
      std::string msg = "qubit index out of range";
      throw std::runtime_error(msg);
    }
  for (auto q : op.clbits)
    if (q >= nclbits) {
      std::string msg = "clbit index out of range";
      throw std::runtime_error(msg);
    }

#ifdef DEBUG
  std::clog << "DEBUG (json): op " << label;
  if (!op.params.empty())
    std::clog << op.params;
  if (!op.qubits.empty())
    std::clog << " qubits = " << op.qubits;
  if (!op.clbits.empty())
    std::clog << " clbits = " << op.clbits;
  std::clog << std::endl;
#endif

  // Load if operation parameters
  if (JSON::check_key("conditional", node)) {
    op.cond = parse_conditional(node["conditional"]);
    op.if_op = true;
  }

  return op;
}

//------------------------------------------------------------------------------
operation_if Circuit::parse_conditional(const json_t &jcond) {
  operation_if cond;
  // Parse json values
  std::string maskstr, valstr;
  JSON::get_value(cond.type, "type", jcond);
  JSON::get_value(maskstr, "mask", jcond);
  JSON::get_value(valstr, "val", jcond);

  // Setup mask
  cond.mask = hex_to_creg(maskstr); // Get mask
  auto pos = std::find(cond.mask.rbegin(), cond.mask.rend(), 1);
  auto sz = std::distance(pos, cond.mask.rend());
  cond.mask.resize(sz); // resize to smallest vector

  // Get target val
  cond.val = hex_to_creg(valstr);
  auto sz_val = std::count(cond.mask.cbegin(), cond.mask.cend(), 1);
  cond.val.resize(sz_val, 0);

#ifdef DEBUG
  std::clog << "DEBUG: if_type = " << cond.type << std::endl;
  std::clog << "DEBUG: if_mask = " << cond.mask << std::endl;
  std::clog << "DEBUG: if_val = " << cond.val << std::endl;
#endif

  return cond;
}

//------------------------------------------------------------------------------
reglist Circuit::parse_reglist(const json_t &node) {
  if (node.is_array() == false) {
    throw std::runtime_error(std::string("invalid reglist"));
  } else {
    reglist regs;
    for (auto it = node.begin(); it != node.end(); ++it)
      if (it->is_array() && it->size() == 2) {
        auto jr = *it;
        regType r{jr[0].get<std::string>(), jr[1].get<uint_t>()};
        regs.push_back(r);
      }
    return regs;
  }
}

//------------------------------------------------------------------------------
bool Circuit::check_opt_meas() {
  // find first instance of a measurement
  uint_t pos = 0;
  while (operations[pos].id != gate_t::Measure && pos < operations.size()) {
    pos++;
  }
  // Check all remaining operations are also measurements
  bool pass = true;
  for (auto it = operations.cbegin() + pos; it < operations.cend(); it++)
    pass &= (it->id == gate_t::Measure);
  return pass;
}

//------------------------------------------------------------------------------
} // end namespace QISKIT
#endif
