# Qiskit Terra
[![License](https://img.shields.io/github/license/Qiskit/qiskit-terra.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)<!--- long-description-skip-begin -->[![Release](https://img.shields.io/github/release/Qiskit/qiskit-terra.svg?style=popout-square)](https://github.com/Qiskit/qiskit-terra/releases)[![Downloads](https://img.shields.io/pypi/dm/qiskit-terra.svg?style=popout-square)](https://pypi.org/project/qiskit-terra/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-terra/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-terra?branch=main)[![Minimum rustc 1.61.0](https://img.shields.io/badge/rustc-1.61.0+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)<!--- long-description-skip-end -->

**Qiskit** is an open-source framework for working with noisy quantum computers at the level of pulses, circuits, and algorithms.

This library is the core component of Qiskit, **Terra**, which contains the building blocks for creating
and working with quantum circuits, programs, and algorithms. It also contains a compiler that supports
different quantum computers and a common interface for running programs on different quantum computer architectures.

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

Now that Qiskit is installed, it's time to begin working with Qiskit. To do this
we create a `QuantumCircuit` object to define a basic quantum program.

```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0,1], [0,1])
```

This simple example makes an entangled state, also called a [Bell state](https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html#3.2-Entangled-States-).

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

For further examples of using Qiskit you can look at the example scripts in **examples/python**. You can start with
[using_qiskit_terra_level_0.py](examples/python/using_qiskit_terra_level_0.py) and working up in the levels. Also
you can refer to the tutorials in the documentation here:

https://qiskit.org/documentation/tutorials.html


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
