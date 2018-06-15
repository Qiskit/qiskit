/**
 * Copyright 2017, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file main.cpp
 * @brief QISKIT Simulator
 * @author Christopher J. Wood <cjwood@us.ibm.com>
 */

//#define DEBUG // Uncomment for verbose debugging output

#include <cstdio>
#include <iostream>
#include <string>

// Simulator
#include "simulator.hpp"

/*******************************************************************************
 *
 * Main
 *
 ******************************************************************************/

inline void failed(std::string msg, std::ostream &o = std::cout,
                   int indent = -1) {
  json_t ret;
  ret["success"] = false;
  ret["status"] = std::string("ERROR: ") + msg;
  o << ret.dump(indent) << std::endl;
}

int main(int argc, char **argv) {

  std::ostream &out = std::cout; // output stream
  int indent = 4;
  json_t qobj;

  // Parse the input from cin or stream
  if (argc == 2) {
    try {
      qobj = JSON::load(std::string(argv[1]));
    } catch (std::exception &e) {
      std::stringstream msg;
      msg << "Invalid input (" << e.what() << ")";
      failed(msg.str(), out, indent);
      return 1;
    }
  } else {
    failed("Invalid command line", out);
    // Print usage message
    std::cerr << std::endl;
    std::cerr << "qsikit_simulator file" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  file : qobj file\n" << std::endl;
    return 1;
  }

  // Execute simulation
  try {
    QISKIT::Simulator sim;
    sim.load_qobj_json(qobj);

// Set qubit limit
#if defined MAX_QUBITS
    sim.max_qubits = std::map<std::string, uint_t>{{"qubit", MAX_QUBITS}};
#endif

    // Execute
    out << sim.execute() << std::endl;
    //out << sim.execute().dump(indent) << std::endl;
    return 0;
  } catch (std::exception &e) {
    std::stringstream msg;
    msg << "Failed to execute qobj (" << e.what() << ")";
    failed(msg.str(), out, indent);
    return 1;
  }

} // end main
