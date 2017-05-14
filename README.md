# Qiskit SDK Python

[![Build Status](https://travis.ibm.com/IBMQuantum/qiskit-sdk-py-dev.svg?token=GMH4xFrA9iezVJKqw2zH&branch=master)](https://travis.ibm.com/IBMQuantum/qiskit-sdk-py-dev)

Python software development kit (SDK) and Jupyter notebooks for working with
OPENQASM and the IBM Quantum Experience (QE).

Related external projects:

- [Python API](https://github.com/IBM/qiskit-api-py)
- [OPENQASM](https://github.com/IBM/qiskit-openqasm)

## Organization

The *tutorial* directory contains Jupyter notebooks showing how to use the
[Python API](https://github.com/IBM/qiskit-api-py) with
[OPENQASM](https://github.com/IBM/qiskit-openqasm).

There are Jupyter notebooks demonstrating components of
the SDK in the *tutorial* directory, and more python and qasm examples in the *examples* directory. There are also command line test scripts
in the *test* directory. The root directory contains some miscellaneous
examples and an index Jupyter notebook.

We want to reorganize the SDK so that it has a comfortable and intuitive
interface for developers.

Users can create instances of *QuantumRegister* and *ClassicalRegister*, and
use these to construct a *QuantumCircuit*. They can then call methods of these
objects to apply gates within the circuit. The *extensions* directory extends
these objects as needed to support new gate sets and algorithms. The "cswap"
gate in the standard extension shows how to build gates that are sequences of
other unitary gates. The Python file "header.py" shows how we append OPENQASM
gate definitions as we import extensions. The *QuantumCircuit* can generate
OPENQASM code that can flow through other components in the *qiskit* directory.

The *qiskit* directory is the main Python module and contains the programming
interface objects *QuantumRegister*, *ClassicalRegister*, and *QuantumCircuit*.
The directory also contains a *qasm* module for parsing OPENQASM circuits,
an *unroll* module to flatten QASM for a target gate basis by expanding
gate subroutines as needed, a *circuit* module for working with circuits as
graphs, and a *localize* module for mapping all-to-all circuits to run on
devices with fixed couplings.

Quantum circuits flow through the components as follows. The programming
interface is used to generate **OPENQASM** circuits. **OPENQASM** source,
as a file or string, is passed into a *Qasm* object, whose *parse* method
produces an abstract syntax tree (**AST**) representation. The **AST** is
passed to an *Unroller* that is attached to an *UnrollerBackend*. There is
a *PrinterBackend* for outputting text and a *CircuitBackend* for constructing *Circuit* objects. The *Circuit* object represents an unrolled **OPENQASM**
circuit as a directed acyclic graph (**DAG**). The *Circuit* provides methods
for representing, transforming, and computing properties of a circuit as a
**DAG** and outputting the results again as **OPENQASM**. The whole flow is
used by the *localize* module's *swap_mapper* method to insert SWAP gates
so a circuit can execute on a device with fixed couplings given by a
*CouplingGraph*.

## Install

- Intall Anaconda: https://www.continuum.io/downloads
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

- Setup the Jupyter notebook. Add your API token to the file "Qconfig.py" (get it from [IBM Quantum Experience](https://quantumexperience.ng.bluemix.net) > Account):

```sh
cp tutorial/Qconfig.py.default Qconfig.py
```

- Run it:

```sh
make run
```

## FAQ

If you upgrade the dependencies and next error happens try with this fix:

```sh
pip install --upgrade IBMQuantumExperience
* Cannot remove entries from nonexistent file [PATH]/easy-install.pth

# Fix
curl https://bootstrap.pypa.io/ez_setup.py -o - | python
```

## Developer guide

Please, use [GitHub pull requests](https://help.github.com/articles/using-pull-requests) to send the contributions.

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

Note: You can get yout "putYourQExperienceTokenHere" from [IBM Quantum Experience](https://quantumexperience.ng.bluemix.net) > Account)

### Commit messages rules

- It should be formed by a one-line subject, followed by one line of white space. Followed by one or more descriptive paragraphs, each separated by one￼￼￼￼ line of white space. All of them finished by a dot.
- If it fixes an issue, it should include a reference to the issue ID in the first line of the commit.
- It should provide enough information for a reviewer to understand the changes and their relation to the rest of the code.
