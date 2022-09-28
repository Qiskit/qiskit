# Qiskit Terra
[![License](https://img.shields.io/github/license/Qiskit/qiskit-terra.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)<!--- long-description-skip-begin -->[![Build Status](https://img.shields.io/travis/com/Qiskit/qiskit-terra/master.svg?style=popout-square)](https://travis-ci.com/Qiskit/qiskit-terra)[![Release](https://img.shields.io/github/release/Qiskit/qiskit-terra.svg?style=popout-square)](https://github.com/Qiskit/qiskit-terra/releases)[![Downloads](https://img.shields.io/pypi/dm/qiskit-terra.svg?style=popout-square)](https://pypi.org/project/qiskit-terra/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-terra/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-terra?branch=main)[![Minimum rustc 1.56.1](https://img.shields.io/badge/rustc-1.56.1+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)<!--- long-description-skip-end -->

**Qiskit** is an open-source framework for working with noisy quantum computers at the level of pulses, circuits, and algorithms.

The core component of Qiskit is **Terra**.

## Installation

We encourage installing Qiskit via the pip tool (a python package manager). The following command installs the core Qiskit components, including Terra.

```bash
pip install qiskit
```

Pip will handle all dependencies automatically and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [documentation](https://qiskit.org/documentation/contributing_to_qiskit.html#install-install-from-source-label).

## Creating Your First Quantum Program in Qiskit Terra

Now that Qiskit is installed, it's time to begin working with Terra.

We are ready to try out a quantum circuit example, which is simulated locally using Qiskit's `BasicAer`. This is a simple example that makes an entangled state.

```
$ python
```

```python
>>> from qiskit import QuantumCircuit, transpile
>>> from qiskit.providers.basicaer import QasmSimulatorPy
>>> qc = QuantumCircuit(2, 2)
>>> qc.h(0)
>>> qc.cx(0, 1)
>>> qc.measure([0,1], [0,1])
>>> backend_sim = QasmSimulatorPy()
>>> transpiled_qc = transpile(qc, backend_sim)
>>> result = backend_sim.run(transpiled_qc).result()
>>> print(result.get_counts(qc))
```

In this case, the output will be:

```python
{'00': 513, '11': 511}
```

### Executing your code on a real quantum chip

You can also use Qiskit to execute your code on a
**real quantum chip**.
Qiskit enables you to run code on real quantum computers. To to do this it provides an abstraction
layer that lets hardware vendors build packages that provde an interface to their hardware from Qiskit.
Using these ``providers`` you can run any Qiskit code against a real quantum computer.

For example, to use Qiskit with IBM Quantum's cloud quantum computing service:

#### Configure your IBM Quantum credentials

1. Create an _[IBM Quantum](https://quantum-computing.ibm.com)_ account if you haven't already done so.

2. Get an API token from the IBM Quantum website under _API Token_.  
<a href="https://user-images.githubusercontent.com/766693/189518233-fa463e4d-08b8-4a0f-b61c-5d5e978acd8b.png"><img src="https://user-images.githubusercontent.com/766693/189518233-fa463e4d-08b8-4a0f-b61c-5d5e978acd8b.png" width="80%"/></a>

3. Take your token from step 2, here called `MY_API_TOKEN`, and run:

   ```python
   >>> from qiskit import IBMQ
   >>> IBMQ.save_account('MY_API_TOKEN')
    ```

After calling `IBMQ.save_account()`, your credentials will be stored persistently in a file.
Once they are stored, at any point in the future you can load and use them
in your program simply via:

```python
>>> from qiskit import IBMQ
>>> IBMQ.load_account()
```

Those who do not want to save their credentials in a persistent file should use instead:

```python
>>> from qiskit import IBMQ
>>> IBMQ.enable_account('MY_API_TOKEN')
```

and the token will only be active for the session. For examples using Terra with real
devices we have provided a set of examples in **examples/python** and we suggest starting with [using_qiskit_terra_level_0.py](examples/python/using_qiskit_terra_level_0.py) and working up in
the levels.

## Contribution Guidelines

If you'd like to contribute to Qiskit Terra, please take a look at our
[contribution guidelines](CONTRIBUTING.md). This project adheres to Qiskit's [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-terra/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
and use our [Qiskit Slack channel](https://qiskit.slack.com) for discussion and simple questions.
For questions that are more suited for a forum we use the `qiskit` tag in the [Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from our
[Qiskit Tutorials](https://github.com/Qiskit/qiskit-tutorials) repository.

## Authors and Citation

Qiskit Terra is the work of [many people](https://github.com/Qiskit/qiskit-terra/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included [BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

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
