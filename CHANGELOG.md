# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

> **Tags:**
> - 🎉 Added
> - ✏️ Changed
> - ⚠️ Deprecated
> - ❌ Removed
> - 🐛 Fixed
> - 👾 Security

## [Unreleased]



## [0.5.3] - 2018-05-29

### 🎉 Added

- load_qasm_file / load_qasm_string methods

### ✏️ Changed

- Dependencies version bumped

### 🐛 Fixed

- Crash in the cpp simulator for some linux platforms
- Fixed some minor bugs

## [0.5.2] - 2018-05-21

### ✏️ Changed

- Adding Result.get_unitary()

### ⚠️ Deprecated

- Deprecating ibmqx_hpc_qasm_simulator and ibmqx_qasm_simulator in favor of: ibmq_qasm_simulator.

### 🐛 Fixed

- Fixing a Mapper issue.
- Fixing Windows 7 builds.

## [0.5.1] - 2018-05-15

There are no code changes.

MacOS simulator has been rebuilt with external user libraries compiled statically, so there's no need for users to have a preinstalled gcc environment.

Pypi forces us to bump up the version number if we want to upload a new package, so this is basically what have changed.

## [0.5.0] - 2018-05-11

⚠️ TODO ⚠️

## [0.4.15] - 2018-05-07

### 🐛 Fixed

- Fixed an issue with legacy code that was affecting Developers Challenge

## [0.4.14] - 2018-04-18

### 🐛 Fixed

- Fixed an issue about handling Basis Gates parameters on backend configurations

## [0.4.13] - 2018-04-16

### ✏️ Changed

- OpenQuantumCompiler.dag2json() restored for backward compatibility

### 🐛 Fixed

- Fixes an issue regarding barrier gate misuse in some circumstances

## [0.4.12] - 2018-03-11

### ✏️ Changed

- Improved circuit visualization.
- Improvements in infrastructure code, mostly tests and build system.
- Better documentation regarding contributors

### 🐛 Fixed

- A bunch of minor bugs have been fixed.

## [0.4.11] - 2018-03-13

### 🎉 Added

- More testing :)

### ✏️ Changed

- Stabilizing code related to external dependencies

### 🐛 Fixed

- Fixed bug in circuit drawing where some gates in the standard library were not plotting correctly

## [0.4.10] - 2018-03-06

### 🎉 Added

- Chinese translation of README

### ✏️ Changed

- Changes related with infrastructure (linter, tests, automation) enhancement

### 🐛 Fixed

- Fix installation issue when simulator cannot be built
- Fix bug with auto-generated CNOT coherent error matrix in C++ simulator
- Fix a bug in the async code

## [0.4.9] - 2018-02-12

### ✏️ Changed

- CMake integration
- QASM improvements
- Mapper optimizer improvements

### 🐛 Fixed

- Some minor C++ Simulator bug-fixes

## [0.4.8] - 2018-01-29

### 🐛 Fixed

- Fix parsing U_error matrix in C++ Simulator python helper class
- Fix display of code-blocks on .rst pages

## [0.4.7] - 2018-01-26

### ✏️ Changed

- Changes some naming conventions for "amp_error" noise parameters to "calibration_error"

### 🐛 Fixed

- Fixes several bugs with noise implementations in the simulator.
- Fixes many spelling mistakes in simulator README.

## [0.4.6] - 2018-01-22

### ✏️ Changed

- We have upgraded some of out external dependencies to:
    - matplotlib >=2.1,<2.2
    - networkx>=1.11,<2.1
    - numpy>=1.13,<1.15
    - ply==3.10
    - scipy>=0.19,<1.1
    - Sphinx>=1.6,<1.7
    - sympy>=1.0

## [0.4.4] - 2018-01-09

### ✏️ Changed

- Update dependencies to more recent versions

### 🐛 Fixed

- Fix bug with process tomography reversing qubit preparation order

## [0.4.3] - 2018-01-08

### ⚠️ Deprecated

- Static compilation has been removed because it seems to be failing while installing Qiskit via pip on Mac.

## [0.4.2] - 2018-01-08

### 🐛 Fixed

- Minor bug fixing related to pip installation process.

## [0.4.0] - 2018-01-08

⚠️ TODO ⚠️

[Unreleased]: https://github.com/QISKit/qiskit-core/compare/0.5.3...HEAD
[0.5.3]: https://github.com/QISKit/qiskit-core/compare/0.5.2...0.5.3
[0.5.2]: https://github.com/QISKit/qiskit-core/compare/0.5.1...0.5.2
[0.5.1]: https://github.com/QISKit/qiskit-core/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/QISKit/qiskit-core/compare/0.4.15...0.5.0
[0.4.15]: https://github.com/QISKit/qiskit-core/compare/0.4.14...0.4.15
[0.4.14]: https://github.com/QISKit/qiskit-core/compare/0.4.13...0.4.14
[0.4.13]: https://github.com/QISKit/qiskit-core/compare/0.4.12...0.4.13
[0.4.12]: https://github.com/QISKit/qiskit-core/compare/0.4.11...0.4.12
[0.4.11]: https://github.com/QISKit/qiskit-core/compare/0.4.10...0.4.11
[0.4.10]: https://github.com/QISKit/qiskit-core/compare/0.4.9...0.4.10
[0.4.9]: https://github.com/QISKit/qiskit-core/compare/0.4.8...0.4.9
[0.4.8]: https://github.com/QISKit/qiskit-core/compare/0.4.7...0.4.8
[0.4.7]: https://github.com/QISKit/qiskit-core/compare/0.4.6...0.4.7
[0.4.6]: https://github.com/QISKit/qiskit-core/compare/0.4.5...0.4.6
[0.4.5]: https://github.com/QISKit/qiskit-core/compare/0.4.4...0.4.5
[0.4.4]: https://github.com/QISKit/qiskit-core/compare/0.4.3...0.4.4
[0.4.3]: https://github.com/QISKit/qiskit-core/compare/0.4.2...0.4.3
[0.4.2]: https://github.com/QISKit/qiskit-core/compare/0.4.1...0.4.2
[0.4.0]: https://github.com/QISKit/qiskit-core/compare/0.3.16...0.4.0
