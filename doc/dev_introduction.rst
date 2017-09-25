Structure
=========

Programming interface
---------------------

The *qiskit* directory is the main Python module and contains the
programming interface objects :py:class:`QuantumProgram <qiskit.QuantumProgram>`,
:py:class:`QuantumRegister <qiskit.QuantumRegister>`,
:py:class:`ClassicalRegister <qiskit.ClassicalRegister>`,
and :py:class:`QuantumCircuit <qiskit.QuantumCircuit>`.

At the highest level, users construct a *QuantumProgram* to create,
modify, compile, and execute a collection of quantum circuits. Each
*QuantumCircuit* has a set of data registers, each of type
*QuantumRegister* or *ClassicalRegister*. Methods of these objects are
used to apply instructions that define the circuit. The *QuantumCircuit*
can then generate **OpenQASM** code that can flow through other
components in the *qiskit* directory.

The :py:mod:`extensions <qiskit.extensions>` directory extends quantum circuits
as needed to support other gate sets and algorithms. Currently there is a
:py:mod:`standard <qiskit.extensions.standard>` extension defining some typical
quantum gates.

Internal modules
----------------

The directory also contains internal modules that are still under development:

- a :py:mod:`qasm <qiskit.qasm>` module for parsing **OpenQASM** circuits
- an :py:mod:`unroll <qiskit.unroll>` module to interpret and “unroll”
  **OpenQASM** to a target gate basis (expanding gate subroutines and loops as
  needed)
- a :py:mod:`dagcircuit <qiskit.dagcircuit>` module for working with circuits as
  graphs
- a :py:mod:`mapper <qiskit.mapper>` module for mapping all-to-all circuits to
  run on devices with fixed couplings
- a :py:mod:`simulators <qiskit.simulators>` module contains quantum circuit
  simulators
- a *tools* directory contains methods for applications, analysis, and visualization

Quantum circuits flow through the components as follows. The programming interface is used to
generate **OpenQASM** circuits, as text or *QuantumCircuit* objects. **OpenQASM** source, as a
file or string, is passed into a *Qasm* object, whose parse method produces an abstract syntax
tree (**AST**). The **AST** is passed to an *Unroller* that is attached to an *UnrollerBackend*.
There is a *PrinterBackend* for outputting text, a *JsonBackend* for producing input to
simulator and experiment backends, a *DAGBackend* for constructing *DAGCircuit* objects, and
a *CircuitBackend* for producing *QuantumCircuit* objects. The *DAGCircuit* object represents
an “unrolled” **OpenQASM** circuit as a directed acyclic graph (DAG). The *DAGCircuit* provides
methods for representing, transforming, and computing properties of a circuit and outputting the
results again as **OpenQASM**. The whole flow is used by the *mapper* module to rewrite a
circuit to execute on a device with fixed couplings given by a *CouplingGraph*. The structure of
these components is subject to change.

The circuit representations and how they are currently transformed into each other are summarized
in this figure:



.. image:: ../images/circuit_representations.png
    :width: 600px
    :align: center

Several unroller backends and their outputs are summarized here:



.. image:: ../images/unroller_backends.png
    :width: 600px
    :align: center
