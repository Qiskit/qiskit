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
in the *testscripts* directory. The root directory contains some miscellaneous
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

- Virtual environment:

```sh
# Creation
conda create -n QISKitenv python=3 pip
# Setup
source activate QISKitenv
```

- Install project dependencies:

```sh
pip install -r requires.txt
```

- Setup the Jupyter notebook. Add your API token to the file "Qconfig.py" (get it from [IBM Quantum Experience](https://quantumexperience.ng.bluemix.net) > Account):

```sh
cd script
mv Qconfig.py.default Qconfig.py
```

- Run it:

```sh
jupyter notebook
```

### Dependencies problem

If you upgrade the dependencies and next error happens try with this fix:

```sh
pip install --upgrade IBMQuantumExperience
* Cannot remove entries from nonexistent file [PATH]/easy-install.pth

# Fix
curl https://bootstrap.pypa.io/ez_setup.py -o - | python
```
