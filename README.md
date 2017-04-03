# Python SDK

Python software development kit (SDK) and Jupyter notebooks for working with
OPENQASM and the IBM Quantum Experience (QE).

Related external projects:

- [Python API](https://github.com/IBM/qiskit-api-py)
- [OPENQASM](https://github.com/IBM/qiskit-openqasm)

## Organization

The *scripts* directory contains Jupyter notebooks showing how to use the
[Python API](https://github.com/IBM/qiskit-api-py) with
[OPENQASM](https://github.com/IBM/qiskit-openqasm).

There are Jupyter notebooks demonstrating components of
the SDK in the *scripts* directory. There are also command line test scripts
in the *testscripts* directory. There are what will eventually be whole-SDK
example scripts in the root directory.

We want to reorganize the SDK so that it has a
comfortable and intuitive interface for developers. I will learn from
Ismael, Fran, Jorge, and Paco what architecture makes sense and try to
implement that architecture. This is meant as a place for us to try ideas
until we settle on something that makes sense for everyone.

Right now, users can create instances of *QuantumRegister* and *ClassicalRegister*, and bind
these to *QuantumCircuit*. They can then call methods of these objects to
apply gates within the circuit. The *extensions* directory extends these
objects as needed to support new gate sets and algorithms. The "cswap" gate in
the standard extension shows how to build gates that are sequences of other
unitary gates. The Python file "header.py" shows how we append OPENQASM gate
definitions as we import extensions.

The *qiskit* directory is the main Python
module. It contains a *qasm* module for parsing OPENQASM circuits,
an *unroll* module to flatten QASM for a target gate basis, a *circuit* module
for representing, transforming, and computing properties of OPENQASM circuits
as directed acyclic graphs, and a *localize* module for mapping all-to-all
circuits to run on machines with fixed couplings.

Quantum circuits flow through the components as follows. **OPENQASM** source,
as a file or string, is passed into a *qasm* object, whose *parse* method
produces an abstract syntax tree (**AST**) representation. The **AST** is
passed to an *Unroller* that is attached to an *UnrollerBackend*. There is
an unimplemented base class, a *PrinterBackend* for outputting text, and
a *CircuitBackend* for constructing *circuit* objects. The *circuit* object
represents an unrolled **OPENQASM** circuit as a directed acyclic graph
(**DAG**). The *circuit* provides methods for working with circuits and
outputting **OPENQASM**.

## Setup Python Virtual Enviroment

### Anaconda
To create a new Virtual Enviroment:
> conda create -n QISKitenv python=3 pip

use it:

> source activate QISKitenv

### Install Dependencies

> pip install -r requires.txt

### Dependencies problems.

When you try to install the dependencies "pip install --upgrade IBMQuantumExperience", if you have the next error:

* Cannot remove entries from nonexistent file [PATH]/easy-install.pth

You can fix it using:

> curl https://bootstrap.pypa.io/ez_setup.py -o - | python
