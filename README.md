# Qiskit Terra
[![License](https://img.shields.io/github/license/Qiskit/qiskit-terra.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)<!--- long-description-skip-begin -->[![Release](https://img.shields.io/github/release/Qiskit/qiskit-terra.svg?style=popout-square)](https://github.com/Qiskit/qiskit-terra/releases)[![Downloads](https://img.shields.io/pypi/dm/qiskit-terra.svg?style=popout-square)](https://pypi.org/project/qiskit-terra/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-terra/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-terra?branch=main)[![Minimum rustc 1.64.0](https://img.shields.io/badge/rustc-1.64.0+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)<!--- long-description-skip-end -->

**Qiskit**  is an open-source SDK for working with quantum computers at the level of extended quantum circuits, operators, and primitives.

This library is the core component of Qiskit, **Terra**, which contains the building blocks for creating
and working with quantum circuits, quantum operators, and primitive functions (sampler and estimator).
It also contains a transpiler that supports optimizing quantum circuits and a quantum information toolbox for creating advanced quantum operators. 

For more details on how to use Qiskit you can refer to the documentation located here:

https://qiskit.org/documentation/


## Installation

We encourage installing Qiskit via ``pip``. The following command installs the core Qiskit components, including Terra.

```bash
pip install qiskit
```

Pip will handle all dependencies automatically and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [documentation](https://qiskit.org/documentation/contributing_to_qiskit.html#install-install-from-source-label).

## Creating Your First Quantum Program in Qiskit Terra

Now that Qiskit is installed, it's time to begin working with Qiskit. The essential parts of a quantum program are 
1. Define and build a quantum circuit that represents the quantum state
2. Define the classical output by measurements or a set of observable operators
3. Depending on the output, use the primitive function `sampler` to sample outcomes or the `estimaor` to estimate values.

Usign the `QuantumCircuit` object a example quantum circuit can be created using:

```python
import numpy as np
from qiskit import QuantumCircuit

# A quantum circuit for preparing the quantum state |000> + i |111>
qc_example = QuantumCircuit(3)
qc_example.h(0) # generate superpostion
qc_example.p(np.pi/2,0) # add quantum phase
qc_example.cx(0,1) # condition 1st qubit on 0th qubit
qc_example.cx(0,2) # condition 2nd qubit on 0th qubit
```

This simple example makes an entangled state known as a GHZ state `(|000> + i |111>)/rt(2)`. It uses the standard quantum 
gates: Hadamard gate, Phase gate and CNOT. 

Once you've made your first quantum circuit, you need to decide on which primtive function you will use. Starting with the sampler 
we use the `compose` funtion to add a measurement circuit to the example circuit. In this example we simply map the qubits to
the classical registers in ascending order. 

```python
qc_measure = QuantumCircuit(3,3)
qc_measure.measure_all(add_bits=False)
qc_compose = qc_example.compose(qc_measure)

from qiskit.primitives.sampler import Sampler
sampler = Sampler()
job = sampler.run(qc_compose, shots=1000)
result = job.result()
print(f" > Quasi probability distribution: {result.quasi_dists}")
```
Running this will give an outcome similar to `{0: 0.497, 7: 0.503}` which is `000` 50% of the time and `111` 50% of the time up to statistical fluctuations.  
To illustrate the power of estimator we now use the quantum information toolbox to create the operator `XXY+XYX+YXX-YYY`.

```python
from qiskit.quantum_info import SparsePauliOp
operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])

from qiskit.primitives.estimator import Estimator
estimator = Estimator()
job = estimator.run(qc_example, operator, shots=1000)
result = job.result()
print(f" > Expectation values: {result.values}")
```

Running this will give the outcome `4`. For fun try to assign a value of +/- 1 to each single qubit operator X and Y 
and see if you can acheive this outcome. This is not possible!

Using the Qiskit provided sampler and estimator will not take you very far. The power of quantum computing cannot be simulated 
on classical computers and you need to use real quantum hardware to scale to larger quantum circuits. However, running a quantum 
circuit on hardware requires rewriting them to the basis gates and connectivity of the quantum hardware.
The tool that does this is the [transpiler](https://qiskit.org/documentation/tutorials/circuits_advanced/04_transpiler_passes_and_passmanager.html) 
and Qiskit includes transpiler passes for synthesis, optimization, mapping, and scheduling. However, it also includes a
default compiler which works very well in most examples and as a example the following code will map the example circuit to the `basis_gates = ['cz', 'sx', 'rz']` and a linear chain of qubits with the `coupling_map =[[0, 1], [1, 2]]`.

```python
from qiskit import transpile
qc_ibm = transpile(qc_example, basis_gates = ['cz', 'sx', 'rz'], coupling_map =[[0, 1], [1, 2]] , optimization_level=3)
```

For further examples of using Qiskit you can look at the example scripts in **examples/python** and the tutorials 
in the documentation here:

https://qiskit.org/documentation/tutorials.html


### Executing your code on real quantum hardware

Qiskit provides an abstraction layer that lets users run quantum circuits on hardware from any vendor that provides a compatible interface. 
The default way for using Qiskit is to have a runtime environment that provides optimal implementations of `sampler` and `estimator` for a given hardware. This runtime may invole using pre and post processing such as optimized transpiler passes with error supression, error mitigation and eventually error correction built in. The runtime must provide a promise to the user that these primitives functions exist

* https://github.com/Qiskit/qiskit-ibm-runtime

However, as Qiskit transitions to the runtime enviroment some hardware is only supported with the ``providers`` interface, however each provider may perform different types of pre and post processing and return outcomes that are vendor defined:

* https://github.com/Qiskit/qiskit-ibmq-provider
* https://github.com/Qiskit-Partners/qiskit-ionq
* https://github.com/Qiskit-Partners/qiskit-aqt-provider
* https://github.com/qiskit-community/qiskit-braket-provider
* https://github.com/qiskit-community/qiskit-quantinuum-provider
* https://github.com/rigetti/qiskit-rigetti

<!-- This is not an exhasutive list, and if you maintain a provider package please feel free to open a PR to add new providers -->

You can refer to the documentation of these packages for further instructions
on how to get access and use these systems.

## Contribution Guidelines

If you'd like to contribute to Qiskit Terra, please take a look at our
[contribution guidelines](CONTRIBUTING.md). This project adheres to Qiskit's [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-terra/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://qisk.it/join-slack)
and use our [Qiskit Slack channel](https://qiskit.slack.com) for discussion and simple questions.
For questions that are more suited for a forum we use the `qiskit` tag in the [Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from our
[Qiskit Tutorials](https://github.com/Qiskit/qiskit-tutorials) repository.

## Authors and Citation

Qiskit Terra is the work of [many people](https://github.com/Qiskit/qiskit-terra/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included [BibTeX file](CITATION.bib).

## Changelog and Release Notes

The changelog for a particular release is dynamically generated and gets
written to the release page on Github for each release. For example, you can
find the page for the `0.9.0` release here:

https://github.com/Qiskit/qiskit-terra/releases/tag/0.9.0

The changelog for the current release can be found in the releases tab:
[![Releases](https://img.shields.io/github/release/Qiskit/qiskit-terra.svg?style=popout-square)](https://github.com/Qiskit/qiskit-terra/releases)
The changelog provides a quick overview of notable changes for a given
release.

Additionally, as part of each release detailed release notes are written to
document in detail what has changed as part of a release. This includes any
documentation on potential breaking changes on upgrade and new features.
For example, you can find the release notes for the `0.9.0` release in the
Qiskit documentation here:

https://qiskit.org/documentation/release_notes.html#terra-0-9

## License

[Apache License 2.0](LICENSE.txt)
