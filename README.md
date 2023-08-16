# Qiskit
[![License](https://img.shields.io/github/license/Qiskit/qiskit-terra.svg?)](https://opensource.org/licenses/Apache-2.0) <!--- long-description-skip-begin -->
[![Release](https://img.shields.io/github/release/Qiskit/qiskit-terra.svg)](https://github.com/Qiskit/qiskit-terra/releases)
[![Downloads](https://img.shields.io/pypi/dm/qiskit-terra.svg)](https://pypi.org/project/qiskit-terra/)
[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-terra/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-terra?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qiskit)
[![Minimum rustc 1.64.0](https://img.shields.io/badge/rustc-1.64.0+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![Downloads](https://pepy.tech/badge/qiskit-terra)](https://pypi.org/project/qiskit-terra/)<!--- long-description-skip-end -->
[![DOI](https://zenodo.org/badge/161550823.svg)](https://zenodo.org/badge/latestdoi/161550823)

**Qiskit** is an open-source framework for working with noisy quantum computers at the level of pulses, circuits, and algorithms.

This framework allows for building, transforming, and visualizing  quantum circuits. It also contains a compiler that supports
different quantum computers and a common interface for running programs on different quantum computer architectures.

For more details on how to use Qiskit you can refer to the documentation located here:

<https://qiskit.org/documentation/>


## Installation

We encourage installing Qiskit via ``pip``:

```bash
pip install qiskit
```

Pip will handle all dependencies automatically and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [documentation](https://qiskit.org/documentation/contributing_to_qiskit.html#install-install-from-source-label).

## Creating Your First Quantum Program in Qiskit

Now that Qiskit is installed, it's time to begin working with Qiskit. To do this
we create a `QuantumCircuit` object to define a basic quantum program.

```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0,1], [0,1])
```

This example makes an entangled state, also called a [Bell state](https://en.wikipedia.org/wiki/Bell_state).

Once you've made your first quantum circuit, you can then simulate it.
To do this, first we need to compile your circuit for the target backend we're going to run
on. In this case we are leveraging the built-in `BasicAer` simulator. However, this
simulator is primarily for testing and is limited in performance and functionality (as the name
implies). You should consider more sophisticated simulators, such as [`qiskit-aer`](https://github.com/Qiskit/qiskit-aer/),
for any real simulation work.

```python
from qiskit import transpile
from qiskit.providers.basicaer import QasmSimulatorPy
backend_sim = QasmSimulatorPy()
transpiled_qc = transpile(qc, backend_sim)
```

After compiling the circuit we can then run this on the ``backend`` object with:

```python
result = backend_sim.run(transpiled_qc).result()
print(result.get_counts(qc))
```

The output from this execution will look similar to this:

```python
{'00': 513, '11': 511}
```

For further examples of using Qiskit you can look at the tutorials in the documentation here:

<https://qiskit.org/documentation/tutorials.html>

### Executing your code on a real quantum chip

You can also use Qiskit to execute your code on a **real quantum processor**.
Qiskit provides an abstraction layer that lets users run quantum circuits on hardware from any
vendor that provides an interface to their systems through Qiskit. Using these ``providers`` you can run any Qiskit code against
real quantum computers. Some examples of published provider packages for running on real hardware are:

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

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](CONTRIBUTING.md). By participating, you are expected to uphold our [code of conduct](CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-terra/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://qisk.it/join-slack) for discussion, comments, and questions.
For questions related to running or using Qiskit, [Stack Overflow has a `qiskit`](https://stackoverflow.com/questions/tagged/qiskit).
For questions on quantum computing with Qiskit, use the `qiskit` tag in the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit) (please, read first the [guidelines on how to ask](https://quantumcomputing.stackexchange.com/help/how-to-ask) in that forum).


## Authors and Citation

Qiskit is the work of [many people](https://github.com/Qiskit/qiskit-terra/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included [BibTeX file](CITATION.bib).

## Changelog and Release Notes

The changelog for a particular release is dynamically generated and gets
written to the release page on Github for each release. For example, you can
find the page for the `0.9.0` release here:

<https://github.com/Qiskit/qiskit-terra/releases/tag/0.9.0>

The changelog for the current release can be found in the releases tab:
[![Releases](https://img.shields.io/github/release/Qiskit/qiskit-terra.svg?style=flat&label=)](https://github.com/Qiskit/qiskit-terra/releases)
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
