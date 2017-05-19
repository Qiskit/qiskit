# Quantum Information Software Kit (QISKit) SDK Python

[![Build Status](https://travis.ibm.com/IBMQuantum/qiskit-sdk-py-dev.svg?token=GMH4xFrA9iezVJKqw2zH&branch=master)](https://travis.ibm.com/IBMQuantum/qiskit-sdk-py-dev)

Python software development kit (SDK) and Jupyter notebooks for working with
OpenQASM and the IBM Q experience (QX).

## Philosophy

The basic concept of our quantum program is an array of quantum circuits. The program workflow consists of three stages: Build, Compile, and Run. Build allows you to make different quantum circuits that represent the problem you are solving; Compile allows you to rewrite them to run on different backends (simulators/real chips of different quantum volumes, sizes, fidelity, etc); and Run launches the jobs. After the jobs have been run, the data is collected. There are methods for putting this data together, depending on the program. This either gives you the answer you wanted or allows you to make a better program for the next instance.

## Organization

The *tutorial* directory contains Jupyter notebooks demonstrating components of the SDK. Take a look at the [index](https://github.ibm.com/IBMQuantum/qiskit-sdk-py-dev/blob/master/tutorial/index.ipynb) to get started. The SDK uses the [Python API](https://github.com/IBM/qiskit-api-py) to interact with the QX and expresses quantum circuits using [OpenQASM](https://github.com/IBM/qiskit-openqasm). Python example programs can be found in the *examples* directory, and test scripts are located in *test*. The *qiskit* directory is the main module of the SDK.

## Structure

### Programming interface

The *qiskit* directory is the main Python module and contains the programming interface objects *QuantumProgram*, *QuantumRegister*, *ClassicalRegister*, and *QuantumCircuit*.

At the highest level, users construct a *QuantumProgram* to create, modify, compile, and execute a collection of quantum circuits. Each *QuantumCircuit* has a set of data registers, each of type *QuantumRegister* or *ClassicalRegister*. Methods of these objects are used to apply instructions that define the circuit. The *QuantumCircuit* can then generate **OpenQASM** code that can flow through other components in the *qiskit* directory.

The *extensions* directory extends quantum circuits as needed to support other gate sets and algorithms. Currently there is a *standard* extension defining some typical quantum gates.

### Internal modules

The directory also contains internal modules:

* a *qasm* module for parsing **OpenQASM** circuits
* an *unroll* module to interpret and "unroll" **OpenQASM** to a target gate basis (expanding gate subroutines and loops as needed)
* a *circuit* module for working with circuits as graphs
* a *mapper* module for mapping all-to-all circuits to run on devices with fixed couplings

Quantum circuits flow through the components as follows. The programming
interface is used to generate **OpenQASM** circuits. **OpenQASM** source,
as a file or string, is passed into a *Qasm* object, whose *parse* method
produces an abstract syntax tree (**AST**). The **AST** is
passed to an *Unroller* that is attached to an *UnrollerBackend*. There is
a *PrinterBackend* for outputting text, a *SimulatorBackend* for outputting simulator input data for the local simulators, and a *CircuitBackend* for constructing *Circuit* objects. The *Circuit* object represents an "unrolled" **OpenQASM**
circuit as a directed acyclic graph (**DAG**). The *Circuit* provides methods
for representing, transforming, and computing properties of a circuit
and outputting the results again as **OpenQASM**. The whole flow is
used by the *mapper* module to rewrite a circuit to execute on a device
with fixed couplings given by a *CouplingGraph*.

The four circuit representations and how they are currently transformed into each other are summarized in this figure:

<img src="images/circuit_representations.png" alt="circuits" width="500"/>

Several unroller backends and their outputs are summarized here:

<img src="images/unroller_backends.png" alt="backends" width="500"/>


## Install

- Install Anaconda: https://www.continuum.io/downloads
- Clone the repo:

```sh
git clone https://github.ibm.com/IBMQuantum/qiskit-sdk-py-dev
cd qiskit-sdk-py-dev
```

- Create the environment with the dependencies:

```sh
make env
```

## Use

- Set up the Jupyter notebook. Add your API token to the file "Qconfig.py" (get it from [IBM Q experience](https://quantumexperience.ng.bluemix.net) > Account):

```sh
cp tutorial/Qconfig.py.default Qconfig.py
```

- Run it:

```sh
make run
```

## FAQ

If you upgrade the dependencies and get an error, try this fix:

```sh
pip install --upgrade IBMQuantumExperience
* Cannot remove entries from nonexistent file [PATH]/easy-install.pth

# Fix
curl https://bootstrap.pypa.io/ez_setup.py -o - | python
```

## Developer Guide

Please use [GitHub pull requests](https://help.github.com/articles/using-pull-requests) to send contributions.

We use [Pylint](https://www.pylint.org) and [PEP 8](https://www.python.org/dev/peps/pep-0008) style guide.


### Dependencies

```sh
make env-dev
```

### Test

- Please run this to be sure your code fits with the style guide and the tests keep passing:

```sh
make test
```

Note: You can get your "putYourQExperienceTokenHere" from [IBM Q experience](https://quantumexperience.ng.bluemix.net) > Account)

### Commit messages rules

- Commit messages should have a one-line subject, followed by one line of white space, followed by one or more descriptive paragraphs, each separated by one line of white space, and all of them ending with a dot.
- If it fixes an issue, it should include a reference to the issue ID in the first line of the commit.
- It should provide enough information for a reviewer to understand the changes and their relation to the rest of the code.


## Authors (alphabetical)

The first release of QISKit was developed by Jim Challenger, Andrew Cross, Ismael Faro, Jay Gambetta, Jesus Perez, and John Smolin.

## License

QISKit is released under the Apache 2 license.
