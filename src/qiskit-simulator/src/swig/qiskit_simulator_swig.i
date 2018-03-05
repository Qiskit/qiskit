%module qiskit_simulator_swig
%include <std_string.i>

// Add necessary symbols to generated header
%{
#include "qiskit_simulator.hpp"
%}

// Process symbols in header
%include "qiskit_simulator.hpp"