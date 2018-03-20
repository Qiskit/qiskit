# Quantum Information Software Kit (QISKit)

[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/QISKit/qiskit-sdk-py.svg?branch=master)](https://travis-ci.org/QISKit/qiskit-sdk-py)

The Quantum Information Software Kit (**QISKit** for short) is a software development kit (SDK) for
working with [OpenQASM](https://github.com/QISKit/qiskit-openqasm) and the
[IBM Q experience (QX)](https://quantumexperience.ng.bluemix.net/).

Use **QISKit** to create quantum computing programs, compile them, and execute them on one of
several backends (online Real quantum processors, online simulators, and local simulators). For
the online backends, QISKit uses our [python API client](https://github.com/QISKit/qiskit-api-py)
to connect to the IBM Q experience.

**We use GitHub issues for tracking requests and bugs. Please see the**
[IBM Q experience community](https://quantumexperience.ng.bluemix.net/qx/community) **for
questions and discussion.**

**If you'd like to contribute to QISKit, please take a look at our**
[contribution guidelines](CONTRIBUTING.rst).

Links to Sections:

* [Installation](#installation)
* [Creating your first Quantum Program](#creating-your-first-quantum-program)
* [More Information](#more-information)
* [Authors](#authors-alphabetical)
* [License](#license)

## Installation

### Dependencies

At least [Python 3.5 or later](https://www.python.org/downloads/) is needed for using QISKit. In
addition, [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) is recommended
for interacting with the tutorials.
For this reason we recommend installing the [Anaconda 3](https://www.continuum.io/downloads)
python distribution, as it comes with all of these dependencies pre-installed.

In addition, a basic understanding of quantum information is very helpful when interacting with
QISKit. If you're new to quantum, start with our
[User Guides](https://github.com/QISKit/ibmqx-user-guides)!

### Installation

We encourage to install QISKit via the PIP tool (a python package manager):

```
    pip install qiskit
```
PIP will handle all dependencies automatically for us and you will always install the latest (and well-tested) version.

PIP package comes with prebuilt binaries for these platforms:

* Linux x86_64
* Darwin
* Win64

If your platform is not in the list, PIP will try to build from the sources at installation time. It will require to have CMake 3.5 or higher pre-installed and at least one of the [build environments supported by CMake](https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html).

If during the installation PIP doesn't succeed to build, don't worry, you will have QISKit installed at the end but you probably couldn't take advantage of some of the high-performance components. Anyway, we always provide a python, not-so-fast alternative as a fallback.


#### Setup your environment

We recommend using python virtual environments to improve your experience. Refer to our
[Environment Setup documentation](doc/install.rst#3.1-Setup-the-environment) for more information.

## Creating your first Quantum Program

Now that the SDK is installed, it's time to begin working with QISKit.

We are ready to try out a quantum circuit example, which runs via the local simulator.

This is a simple example that makes an entangled state.

```python
from qiskit import QuantumProgram, QISKitError, RegisterSizeError

# Create a QuantumProgram object instance.
q_program = QuantumProgram()
backend = 'local_qasm_simulator'
try:
    # Create a Quantum Register called "qr" with 2 qubits.
    quantum_reg = q_program.create_quantum_register("qr", 2)
    # Create a Classical Register called "cr" with 2 bits.
    classical_reg = q_program.create_classical_register("cr", 2)
    # Create a Quantum Circuit called "qc" involving the Quantum Register "qr"
    # and the Classical Register "cr".
    quantum_circuit =
        q_program.create_circuit("bell", [quantum_reg],[classical_reg])

    # Add the H gate in the Qubit 0, putting this qubit in superposition.
    quantum_circuit.h(quantum_reg[0])
    # Add the CX gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state
    quantum_circuit.cx(quantum_reg[0], quantum_reg[1])

    # Add a Measure gate to see the state.
    quantum_circuit.measure(quantum_reg, classical_reg)

    # Compile and execute the Quantum Program in the local_qasm_simulator.
    result = q_program.execute(["bell"], backend=backend, shots=1024, seed=1)

    # Show the results.
    print(result)
    print(result.get_data("bell"))

except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
except RegisterSizeError as ex:
    print('Error in the number of registers!. Error = {}'.format(ex))
```

In this case, the output will be:

```
COMPLETED
{'counts': {'00': 512, '11': 512}}
```

This script is avaiable [here](examples/python/hello_quantum.py).

### Executing your code on a real Quantum chip

You can also use QISKit to execute your code on a
[real Quantum Chip](https://github.com/QISKit/ibmqx-backend-information).
In order to do so, you need to configure the SDK for using the credentials for
your Quantum Experience Account:


#### Configure your API token and QE credentials


1. Create an [IBM Q experience](https://quantumexperience.ng.bluemix.net)>
   account if you haven't already done so
2. Get an API token from the IBM Q experience website under "`My Account`" >
   "`Personal Access Token`". This API token allows you to execute your
   programs with the IBM Q experience backends.
   [Example](doc/example_real_backend.rst).
3. We are going to create a new file called `Qconfig.py` and insert the API token into it. This file must have these contents:
```python
APItoken = 'MY_API_TOKEN'

config = {
    'url': 'https://quantumexperience.ng.bluemix.net/api',
    # The following should only be needed for IBM Q users.
    'hub': 'MY_HUB',
    'group': 'MY_GROUP',
    'project': 'MY_PROJECT'
}
```
4. Substitute `'MY_API_TOKEN'` with your real API Token extracted in step 2.

5. If you have access to the IBM Q features, you also need to setup the
   values for your hub, group, and project. You can do so by filling the
   `config` variable with the values you can find on your IBM Q account
   page.

Once the `Qconfig.py` file is set up, you have to move it under the same directory/folder where your program/tutorial resides, so it can be imported and be used to authenticate with `QuantumProgram.set_api()` function. For example:

```python
from qiskit import QuantumProgram
import Qconfig

# Creating Programs create your first QuantumProgram object instance.
Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"], verify=False,
                  hub=Qconfig.config["hub"],
                  group=Qconfig.config["group"],
                  project=Qconfig.config["project"])
```

For more details on this and more information see
[our QISKit documentation](https://www.qiskit.org/documentation/).


### Next Steps

Now you're set up and ready to check out some of the other examples from our
[Tutorial](https://github.com/QISKit/qiskit-tutorial) repository. Start with the
[index tutorial](https://github.com/QISKit/qiskit-tutorial/blob/master/index.ipynb) and then go to
the [‘Getting Started’ example](https://github.com/QISKit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb).
If you already have [Jupyter Notebooks installed](https://jupyter.readthedocs.io/en/latest/install.html),
you can copy and modify the notebooks to create your own experiments.

To install the tutorials as part of the QISKit SDK, see the following
[installation details](doc/install.rst#Install-Jupyter-based-tutorials). Complete SDK
documentation can be found in the [*doc* directory](doc/qiskit.rst) and in
[the official QISKit site](https://www.qiskit.org/documentation).

## More Information

For more information on how to use QISKit, tutorial examples, and other helpful links, take a look
at these resources:

* **[User Guides](https://github.com/QISKit/ibmqx-user-guides)**,
  a good starting place for learning about quantum information and computing
* **[Tutorials](https://github.com/QISKit/qiskit-tutorial)**,
  for example notebooks, start with the [index](https://github.com/QISKit/qiskit-tutorial/blob/master/index.ipynb) and [‘Getting Started’ Jupyter notebook](https://github.com/QISKit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)
* **[OpenQASM](https://github.com/QISKit/openqasm)**,
  for additional information and examples of QASM code
* **[IBM Quantum Experience Composer](https://quantumexperience.ng.bluemix.net/qx/editor)**,
  a GUI for interacting with real and simulated quantum computers
* **[QISkit Python API](https://github.com/QISKit/qiskit-api-py)**, an API to use the IBM Quantum
  Experience in Python

QISKit was originally developed by researchers and developers on the
[IBM-Q](http://www.research.ibm.com/ibm-q/) Team at [IBM Research](http://www.research.ibm.com/),
with the aim of offering a high level development kit to work with quantum computers.

Visit the [IBM Q experience community](https://quantumexperience.ng.bluemix.net/qx/community) for
questions and discussions on QISKit and quantum computing more broadly. If you'd like to
contribute to QISKit, please take a look at our [contribution guidelines](CONTRIBUTING.rst).

## Multilanguage guide

* **[Korean Translation](doc/ko/README.md)**, Basic guide line written in Korean.
* **[Chinese Translation](doc/zh/README.md)**, Basic guide line written in Chinese.

## Authors (alphabetical)

Ismail Yunus Akhalwaya, Luciano Bello, Jim Challenger, Andrew Cross, Vincent Dwyer, Mark Everitt, Ismael Faro,
Jay Gambetta, Juan Gomez, Yunho Maeng, Paco Martin, Antonio Mezzacapo, Diego Moreda, Jesus Perez,
Russell Rundle, Todd Tilma, John Smolin, Erick Winston, Chris Wood.

In future releases, anyone who contributes with code to this project is welcome to include their
name here.

## License

This project uses the [Apache License Version 2.0 software license](https://www.apache.org/licenses/LICENSE-2.0).
