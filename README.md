# Quantum Information Science Kit (Qiskit)

[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/Qiskit/qiskit-terra.svg?branch=master)](https://travis-ci.org/Qiskit/qiskit-terra)
[![Build Status IBM Q](https://travis-matrix-badges.herokuapp.com/repos/Qiskit/qiskit-terra/branches/master/8)](https://travis-ci.org/Qiskit/qiskit-terra)

The Quantum Information Science Kit (**Qiskit** for short) is a software development kit (SDK) for
working with [OpenQASM](https://github.com/Qiskit/qiskit-openqasm) and the
[IBM Q Experience (QX)](https://quantumexperience.ng.bluemix.net/).

Use **Qiskit** to create quantum computing programs, compile them, and execute them on one of
several backends (online Real quantum processors, online simulators, and local simulators). For
the online backends, Qiskit uses our [python API client](https://github.com/Qiskit/qiskit-api-py)
to connect to the IBM Q Experience.

**We use GitHub issues for tracking requests and bugs. Please see the**
[IBM Q Experience community](https://quantumexperience.ng.bluemix.net/qx/community) **for
questions and discussion.**

**If you'd like to contribute to Qiskit, please take a look at our**
[contribution guidelines](.github/CONTRIBUTING.rst).

Links to Sections:

* [Installation](#installation)
* [Creating your first Quantum Program](#creating-your-first-quantum-program)
* [More Information](#more-information)
* [Authors](#authors-alphabetical)

## Installation

### Dependencies

At least [Python 3.5 or later](https://www.python.org/downloads/) is needed for using Qiskit. In
addition, [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) is recommended
for interacting with the tutorials.
For this reason we recommend installing the [Anaconda 3](https://www.continuum.io/downloads)
python distribution, as it comes with all of these dependencies pre-installed.

In addition, a basic understanding of quantum information is very helpful when interacting with
Qiskit. If you're new to quantum, start with our
[User Guides](https://github.com/Qiskit/ibmqx-user-guides)!

### Instructions

We encourage to install Qiskit via the PIP tool (a python package manager):

```bash
pip install qiskit
```

PIP will handle all dependencies automatically for us and you will always install the latest (and well-tested) version.

PIP package comes with prebuilt binaries for these platforms:

* Linux x86_64
* Darwin
* Win64

If your platform is not in the list, PIP will try to build from the sources at installation time. It will require to have CMake 3.5 or higher pre-installed and at least one of the [build environments supported by CMake](https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html).

If during the installation PIP doesn't succeed to build, don't worry, you will have Qiskit installed at the end but you probably couldn't take advantage of some of the high-performance components. Anyway, we always provide a python, not-so-fast alternative as a fallback.

#### Setup your environment

We recommend using python virtual environments to improve your experience. Refer to our
[Environment Setup documentation](doc/install.rst#3.1-Setup-the-environment) for more information.

## Creating your first Quantum Program

Now that the SDK is installed, it's time to begin working with Qiskit.

We are ready to try out a quantum circuit example, which runs via the local simulator.

This is a simple example that makes an entangled state.

```python
# Import the Qiskit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import available_backends, execute

# Create a Quantum Register with 2 qubits.
q = QuantumRegister(2)
# Create a Classical Register with 2 bits.
c = ClassicalRegister(2)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)

# Add a H gate on qubit 0, putting this qubit in superposition.
qc.h(q[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qc.cx(q[0], q[1])
# Add a Measure gate to see the state.
qc.measure(q, c)

# See a list of available local simulators
print("Local backends: ", available_backends({'local': True}))

# Compile and run the Quantum circuit on a simulator backend
job_sim = execute(qc, "local_qasm_simulator")
sim_result = job_sim.result()

# Show the results
print("simulation: ", sim_result)
print(sim_result.get_counts(qc))
```

In this case, the output will be:

```python
COMPLETED
{'counts': {'00': 512, '11': 512}}
```

This script is available [here](examples/python/hello_quantum.py), where we also show how to
run the same program on a real quantum computer.

### Executing your code on a real quantum chip

You can also use Qiskit to execute your code on a
[real quantum chip](https://github.com/Qiskit/ibmqx-backend-information).
In order to do so, you need to configure the SDK for using the credentials in
your IBM Q Experience account:

#### Configure your API token and QX credentials

1. Create an _[IBM Q Experience](https://quantumexperience.ng.bluemix.net) > Account_ if you haven't already done so.

2. Get an API token from the IBM Q Experience website under _My Account > Advanced > API Token_. This API token allows you to execute your programs with the IBM Q Experience backends. See: [Example](doc/example_real_backend.rst).

3. We are now going to add the necessary credentials to Qiskit. Take your token
   from step 2, here called `MY_API_TOKEN`, and pass it to the
   `IBMQ.save_account()` function:

   ```python
   from qiskit import IBMQ

   IBMQ.save_account('MY_API_TOKEN')
    ```

4. If you have access to the IBM Q Network features, you also need to pass the
   url listed on your IBM Q account page to `save_account`.

After calling `IBMQ.save_account()`, your credentials will be stored into disk.
Once they are stored, at any point in the future you can load and use them
in your program simply via:

```python
from qiskit import IBMQ

IBMQ.load_accounts()
```

For more details on installing Qiskit and for alternative methods for passing
the IBM QX credentials, such as using environment variables, sending them
explicitly and support for the `Qconfig.py` method available in previous
versions, please check
[our Qiskit documentation](https://www.qiskit.org/documentation/).

### Next Steps

Now you're set up and ready to check out some of the other examples from our
[Tutorial](https://github.com/Qiskit/qiskit-tutorial) repository. Start with the
[index tutorial](https://github.com/Qiskit/qiskit-tutorial/blob/master/index.ipynb) and then go to
the [‘Getting Started’ example](https://github.com/Qiskit/qiskit-tutorial/blob/master/reference/tools/getting_started.ipynb).
If you already have [Jupyter Notebooks installed](https://jupyter.readthedocs.io/en/latest/install.html),
you can copy and modify the notebooks to create your own experiments.

To install the tutorials as part of the Qiskit SDK, see the following
[installation details](doc/install.rst#Install-Jupyter-based-tutorials). Complete SDK
documentation can be found in the [*doc* directory](doc/qiskit.rst) and in
[the official Qiskit site](https://www.qiskit.org/documentation).

## More Information

For more information on how to use Qiskit, tutorial examples, and other helpful links, take a look
at these resources:

* **[User Guides](https://github.com/Qiskit/ibmqx-user-guides)**,
  a good starting place for learning about quantum information and computing
* **[Tutorials](https://github.com/Qiskit/qiskit-tutorial)**,
  for example notebooks, start with the [index](https://github.com/Qiskit/qiskit-tutorial/blob/master/index.ipynb) and [‘Getting Started’ Jupyter notebook](https://github.com/Qiskit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)
* **[OpenQASM](https://github.com/Qiskit/openqasm)**,
  for additional information and examples of QASM code
* **[IBM Quantum Experience Composer](https://quantumexperience.ng.bluemix.net/qx/editor)**,
  a GUI for interacting with real and simulated quantum computers
* **[Qiskit Python API](https://github.com/Qiskit/qiskit-api-py)**, an API to use the IBM Quantum
  Experience in Python

Qiskit was originally developed by researchers and developers on the
[IBM-Q](http://www.research.ibm.com/ibm-q/) Team at [IBM Research](http://www.research.ibm.com/),
with the aim of offering a high level development kit to work with quantum computers.

Visit the [IBM Q Experience community](https://quantumexperience.ng.bluemix.net/qx/community) for
questions and discussions on Qiskit and quantum computing more broadly. If you'd like to
contribute to Qiskit, please take a look at our [contribution guidelines](.github/CONTRIBUTING.rst).

## Multilanguage guide

* **[Korean Translation](doc/ko/README.md)** - basic guide line written in Korean.
* **[Chinese Translation](doc/zh/README.md)** - basic guide line written in Chinese.

## Authors (alphabetical)

Qiskit was originally authored by
Luciano Bello, Jim Challenger, Andrew Cross, Ismael Faro, Jay Gambetta, Juan Gomez,
Ali Javadi-Abhari, Paco Martin, Diego Moreda, Jesus Perez, Erick Winston and Chris Wood.

And continues to grow with the help and work of [many people](https://github.com/Qiskit/qiskit-terra/graphs/contributors) who contribute
to the project at different levels.
