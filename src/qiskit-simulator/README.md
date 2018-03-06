# QISKit Simulator
## A C++ quantum circuit simulator with realistic noise

Copyright (c) 2017 IBM Corporation. All Rights Reserved.

### Authors
 * Christopher J. Wood (<cjwood@us.ibm.com>)
 * John A. Smolin (<smolin@us.ibm.com>)

### Description

*QISKit Simulator* is quantum circuit simulator written in C++ that includes a variety of realistic circuit level noise models. The simulator may be run as a command line application to evaluate quantum programs specified by a *Quantum program OBJect (__QObj__)*. It may also be used as a local backend in the *Quantum Information Software Kit ([__QISKit__](https://www.qiskit.org))* [Python SDK](https://github.com/QISKit/qiskit-sdk-py).

## Contents

- [Installation](#installation)
- [Using the simulator](#using-the-simulator)
    - [Running from the command line](#running-from-the-command-line)
    - [Running in Python](#running-in-python)
    - [Running as a backend for qiskit-sdk-py](#running-as-a-backend-for-qiskit-sdk-py)
    - [Simulator output](#simulator-output)
- [Config Settings](#config-settings)
     - [Using parallelization](#using-parallelization)
    - [Using a custom initial state](#using-a-custom-initial-state)
    - [Output data options](#output-data-options)
- [Noise parameters](#noise-parameters)
    - [Gate Errors](#gate-errors)
    - [Thermal Relaxation Error](#thermal-relaxation-error)
    - [Reset Error](#reset-error)
    - [Measurement Readout Error](#measurement-readout-error)
- [Full config specification](#full-config-specification)
- [Acknowledgements]($#acknowledgements)
- [License](#license)

## Installation

After installing the required dependencies, build the simulator by running `make` in the root directory. This will build the executable `local_qiskit_simulator` in the repository directory. For detailed OS-specific instructions on installing dependencies see the following sections

### Dependencies
Building requires a compiler compatible with the C++14 standard. For example:

- GCC >= 5.1
- Clang >= 3.3

The following packages are required for building:

- BLAS
- LAPACK
- pthread
- OpenMP (optional)

### OS Specific Instructions
#### Installation on MacOS

The simplest way to install is without full OpenMP parallelization. This only requires XCode command line tools. To install run:

```bash
> xcode-select --install
> make
```

To make full use of the parallelization requires OpenMP which requires installing GCC. This can be done by installing the [Homebrew](https://brew.sh/) package manager:

```bash
> /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
> brew install gcc
> make
```

#### Installation on Ubuntu 16.04 LTS

Check that the build-essential, BLAS and LAPACK packages are installed, then make the repository:

```bash
> sudo apt-get update
> sudo apt-get install build-essential libblas-dev liblapack-dev
> make
```

#### Installation on RHEL 6

The default GCC compiler on RHEL 6 does not support C++11. To build you must install an appropriate developer toolset

```bash
> sudo yum install devtoolset-6 blas blas-devel lapack lapack-devel
> scl enable devtoolset-6 bash
> make
```

## Using the simulator


### Running from the command line

After building, the simulator may be run as follows:

```bash
./local_qiskit_simulator input.json
```

where `input.json` is the file name of a **Quantum program object (qobj)**. This is a JSON specification of a quantum program which can be produced using the QISKit SDK (link to qobj specification to come...).

It is also possible to pipe the contents of a qobj directly to the simulator by using replacing the file name with a dash `-`. For example:

```bash
cat input.json | ./local_qiskit_simulator -
```


### Running in Python

The simulator may be called from Python 3 by importing `qiskit/backends/_qiskit_cpp_simulator.py` module. Execution is handled by calling the compiled simulator as a Python subprocess.

```python
# Set the path and file for the simulator executable
SIM_EXECUTABLE = '/path/to/simulator/executable/qiskit_simulator'

# Import simulator
import qiskit.backends._qiskit_cpp_simulator as qs

# Run a qobj on the simulator
qobj = {...}  # qobj as a Python dictionary
result = qs.run(qobj, path=SIM_EXECUTABLE)  # result json as a Python dictionary
```


### Running as a backend for qiskit-sdk-py

This simulator can also be used as a backend for the QISKit Python SDK.  This is handled automatically when importing the `qiskit` module. After importing the module the simulator may be run as follows:

```python
backend = 'local_qiskit_simulator'
shots = <int>
config = <dict>
results = QuantumProgram.execute(circs,
                backend=backend,
                shots=shots,
                config=config)
```

You can check the backend was successfully added using the `available_backends` method of the `QuantumProgram` class. If successful the returned list will include `local_qiskit_simulator` and `local_clifford_simulator`.


### Simulator output

By default the simulator prints a JSON file to the standard output. If called through QISKit this output is loaded and parsed as a Python dictionary. An example simulator output is for a qobj containing a single circuit is:

```python
{
    "backend": "lcoal_qiskit_simulator",
    "data": [{
            "counts": {
                "00": 523,
                "11": 501
            },
            "seed": 3792116984,
            "shots": 1024,
            "time_taken": 0.004356
        }],
    "id": "example_qobj",
    "simulator_backend": "qubit",
    "status": "DONE",
    "time_taken": 0.004366
}
```

The `"data"` key is a list of the output of each circuit in the qobj: If the qobj contains *n* circuits, `"data"` will be a length *n* list. Additional simulation data may be returned by using config settings discussed in the config settings section.

#### qiskit-sdk-py output

If the simulator is called though the Python QISKit SDK the input qobj will only contain a single circuit, and only the values in the `"data"` field will be accessible in the `Results` object. The dictionary of this data may be accessed using the `Results.get_data('name')` method.

#### Complex number format in JSON

In the raw output file, complex numbers are stored as a list of the real and imaginary parts. Eg. for *z = a +ib* the output of *z* will be `[a, b]`. Using this convention complex vectors and matrices are stored as one would expect: as lists of complex numbers, and as lists of lists of complex numbers respectively.

If the simulator is called through the Python qiskit SDK then these are parsed into standard Python complex datatypes.


#### Runtime errors

If the simulator encounters an error at runtime it will return the error message in the JSON file. If the simulator is unable to parse or evaluate the input qobj file it will return

```python
{
    "status": "FAILED",
    "message": "error message"
}
```

If a qobj contains multiple circuits failure of one circuit will not terminate evaluation of the remaining circuits. In this case, the output will appear as

```python
{
    "id": ...,
    "config": ...,
    "time_taken": ...,
    "results": [
        <circ-1 results>, <circ-2 results>, ...
    ]
    "status": "DONE"
}
```

where for any circuit that encounted an error `<circ-n result>` will be given by `{"status": "FAILED", "message": "error message"}`.


## Config Settings

The following table lists all recognized keys and allowed values for the config settings of the input qobj. Any of these options may be specified at the **qobj** config level, or at the individual **circuit** config level. In the latter case, any circuit-level config settings will override the top level config settings.

##### Table of config options
| Key | Allowed Values | Default | Description |
| --- | --- | --- | --- |
| `"shots"`| Integer > 0 | 1 | The number of simulation shots to be evaluated for a given circuit.
| `"seed"` | Integer >= 0 |Random | The initial seed to be used for setting the random number generator used by the simulator. Simulator output should be deterministic of fixed `"seed"` and `"shots_threads"` values.
| `"shots_threads"` | Integer >= 0 | 1 | Allows parallelization of the simulation by evaluating multiple shots simultaneously. Note that this will cause the simulator to use more memory.
| `"data"` | List of strings | None | This control what output data is reported after the simulation. It is a list of string options which are specified in the Data table.
| `"noise_params"` | dict | None | This is a dictionary of noise parameters for the simulation. The allowed noise parameters are specified in the Noise Parameters table.
| `"initial_state"` | Quantum state | None | Allows the circuit to be initialized in a fixed initial state. See the appropriate section for details.
|`"target_states"` | List of quantum states | None | Specifies a list of target quantum states for comparison with the final simulator state if the `"inner_products"` or `"overlaps"` `"data"` options are used. See the appropriate section for details.
| `"renorm_target_states"` | True | Bool |  This option renormalizes all states in the `"target_states`" list to be valid quantum states (with norm 1). If set to `False` the target states will be used as input without normalization.
| `"chop"` | double >= 0 | 1e-10 | Any numerical quantities smaller than this value will be set to zero in the returned output data.  |
| `"max_memory"` | int | 16 | Specifies the maximum memory the simulator should use for storing the state vector. This is used in determining the maximum number of qubits for simulation, and the number of shots to be evaluated in parallel. |
| `"max_threads_shot"` | int | Number of CPU cores | This option may be used to limit the number of shot threads that can be evaluated in parallel. |
| `"max_threads_gate"` | int | Number of CPU cores/shots threads| This option may be used to limit the number of parallel threads that should be used in updating the state vector when performing the state vector update from quantum circuit operations.
| `"threshold_omp_gate"` | int | 20 | This options specifies the qubit number threshold for enabling parallelization when performing the state vector update from quantum circuit operations.

### Maximum qubit number

The maximum qubit number is determined by the `"max_memory"` config setting by using this value as an upper bound on ab estimate of the memory requirements for storing the state vector of an *N* qubit system. This limit is given by the largest *N* such that *16 \* 2<sup>N</sup> \* 10<sup>-9</sup> <* `"max_memory"`. These values are:

| Qubits | Memory (GB)    | Qubits     | Memory (GB) |
| ---    | ---           | ---         | ---         |
| 25     | 0.54          | 31         | 34.36       |
| 26     | 1.07          | 32      | 68.72       |
| 27     | 2.15             | 33         | 137         |
| 28     | 4.29          | 34         | 275         |
| 29     | 8.59          | 35         | 550         |
| 30     | 17.18         | 36      | 1100

### Using parallelization

If compiled with OpenMP support the simulator can use parallelization for both the number of shots evaluated concurrently and for using parallel threads to update the state vector when applying circuit operations. If OpenMP support is not available (for example if compiled using XCode clang on MacOS), then parallelization over shots is still available using the C++11 standard library.

There are two types of parallel speedups available:

- Multithreaded parallel evaluation of shots (OpenMP and C++11)
- Multithreaded parallel update of the state vector for gates and measurements (OpenMP only)

If M1 threads are used in parallel shot evaluation, and M2 threads are used in parallel gate updates, then the total number of threads used is M = M1 * M2.

The total number of threads used is always limited by the available number of CPU cores on a system and is additionally controlled by several other heuristics which will be discussed below. These may be restricted further using the following configuration options: `"max_memory"`, `"max_threads_shot"`, `"max_threads_gate"` `"theshold_omp_gate"`.

#### Parallel evaluation of shots

If multiple shots are being simulated parallel evaluation of shots takes precedence in the used of CPU threads. The total number of threads used is limited by the `"max_memory"` config option and the number of qubits in the circuit. For a given estimate of the memory requirements of a N-qubit state, a number of shot threads will be launched up to the lower number of: the `"max_memory"` limit, the number of shots, the number of CPU cores, the "`max_threads_shot"` config setting.

#### Parallel state vector update

The second type of parallelization is used to update large N-qubit state vectors in parallel. This is only available if the simulator is compiled with **OpenMP** using the `-fopenmp` option. Parallelization is activated when the number of qubits in a circuit is greater than the number specified by `"theshold_omp_gate"`, and it uses any remaining threads *after* shot parallelization. Once above the threshold the number of threads used *per shot thread* is given by the minimum of: the number of CPU cores/number of shot threads (rounded down), the `"max_threads_gate"` config setting. The default threshold is 20 qubits. Lowering this may reduce performance due to the overhead of thread management on the shared state vector.



### Using a custom initial state

By default, the simulator will always be initialized with all qubits in the 0-state. A custom initial state for each shot of the simulator may be specified by using the config setting `"initial_state"`. This maybe be specified in the QOBJ as a vector or in a Bra-Ket style notation. If the initial state is the wrong dimension for the circuit being evaluated then the simulation will fail and return an error message.


##### Qobj Example
The following are all valid representations of the state *|psi> = (|00> + |11> )/sqrt(2)*


- `"initial_state": [0.707107, 0, 0, 0.707107]`
- `"initial_state": [[0.707107, 0], [0,0], [0,0], [0.707107,0]]`
- `"initial_state": {"00": 0.707107, "11": 0.707107}`
- `"initial_state": {"00": [0.707107, 0], "11": [0.707107, 0]}`

The input will be renormalized by the simulator to ensure it is a quantum state. Hence there is no difference between replacing the above inputs with `"initial_state": [1, 0, 0, 1]"`.


##### QISKit Example
When calling the simulator though the QISKit Python SDK the input state may also be a a NumPy array, for example:

```python
config = {'initial_state': np.array([1, 0, 0, 1j]) / np.sqrt(2)}
```

### Output data options

##### Table of config options
| Key |  Description |
| --- | --- |
| `"classical_state"` | Returns a list of the final classical register bitstring after each shot.
| `"quantum_state"`| Returns a list of the final quantum state vector of the system after each shot.
| `"quantum_state_ket"`| As above but with the states represented in ket form.
| `"density_matrix"`| Returns the final quantum state density matrix obtained from averaging the final state vector overall shots.
| `"probabilities"` | Returns the diagonal of the final state density matrix.
| `"probabilities_ket"` | As above but with the probabilities represented in ket form.
| `"target_states_inner"` | Returns a list of the inner products of the final quantum state with the list of target states for each shot. The target states are specified by `"target_states"` config option.
| `"target_states_probs"` | Returns the expectation value of the final quantum state with the list of target states averaged over all shots. The target states are specified by `"target_states"` config option.
| `"saved_quantum_states"` | Returns a list of dictionaries of any saved quantum states for each shot. The state of the system can be saved by the custom `save(n)` gate. the integer `n` will be the key for accessing the saved state in the returned dictionary.
| `"saved_quantum_states_ket"` | As above but with the states represented in ket form.
| `"saved_density_matrix"` | Returns a dictionary of the saved quantum state density matrix obtained by averaging the saved state vector over all shots.
| `"saved_probabilities"` | As above but only returns the diagonal of the density matrix.
| `"probabilities_ket"` | As above but with the probabilities represented in ket form.
| `"saved_target_states_inner"` | Returns a list of the inner products of the saved quantum states with the list of target states for each shot. The target states are specified by `"target_states"` config option.
| `"saved_target_states_probs"` | Returns the expectation value of the saved quantum states with the list of target states averaged over all shots. The target states are specified by `"target_states"` config option.


## Noise Parameters

We now describe the noise model parameters used by the simulator

### Gate Errors

Gate errors are specified by the error name and a dictionary of error parameters. Gate names are

| Name | Operations Affected |
| --- | --- |
| `"id"` | `id` |
| `"CX"` | `CX, cx` |
| `"measure"` | `measure` |
| `"reset"` | `reset` |
| `"U"` | `U, u0, u1, u2, u3, x, y, z, h, s, sdg, t, tdg` |
| `"X90"` | `U, u0, u1, u2, u3, x, y, z, h, s, sdg, t, tdg` |

Note that `"U"` and `"X90"` implement different error models. `"U"` specifies a single qubit error model for all single qubit gates, while `"X90"` specifies an error model for 90-degree X rotation pulses, and single qubit gates are implemented in terms of noisy X-90 pulses and ideal Z-rotations. If both `"U"` and `"X90"` are set, then `"U"` will *only* effect `U` operations, while `"X90"` will affect all other operations (`u0, u1, u2, u3, x, y, z, h, s, sdg, t, tdg`).

In terms of X90 pulses single qubit gates are affected as:

- `u1, z, s, sdg, t, tdg` have no error (zero X-90 pulses)
- `u2, h`: have single gate error (one X-90 pulse)
- `U, u3, x, y` have double gate error (two X-90 pulses)
- `u0`: has multiples of X-90 pulse relaxation error only

The following keys specifify the implemented error models for single qubit gates:

| Key | Values | Description |
| --- | --- | --- |
| `"p_depol"` | p >= 0 | Depolarizing error channel with depolarizing probability *p* |
| `"p_pauli"` | list[p >= 0] | Pauli error channel where the list specifies the Pauli error probabilities. Note that this list will be renormalized to a probability vector. For 1-qubit operations it is `[pI, pX, pY, pZ]`, for 2-qubit operations it is `[pII, pIX, pIY, pIZ, pXI, pXX, .... , pZZ]`. |
| `"gate_time"` | t >=0  | The length of the gate. This is used for computing the thermal relaxation error probability in combination with the `"relaxation_rate"` parameter for thermal relaxation errors. Thermal relaxation is implemented as *T<sub>1</sub>* and *T<sub>2</sub>* relaxation with *T<sub>2</sub> = T<sub>1</sub>*.
| `"U_error"` | unitary matrix | This is a coherent error which is applied after the ideal gate operation.


##### Example

A single qubit gate error with gate time of *1* unit, depolarizing probability *p = 0.001*, dephasing Pauli channel with dephasing probability *pZ = 0.01*, and a coherent phase error *exp(i 0.1)*

```python
"U": {
    "p_depol": 0.001,
    "p_pauli": [0.99, 0, 0, 0.01],
    "gate_time": 1,
    "U_error": [
        [[1, 0], [0, 0]],
        [[0, 0], [0.995004165, 0.099833417]]
    ]
}
```

#### Special Options for X90 and CX coherent errors

The CX and X90 gate have special keys for automatically generating coherent error matrices. This is not supported directly by the simulator, but is handled by the QISKit backend in python.

##### X90 Gate

A coherent error model for X-90 rotations due to calibration errors in the control pulse amplitude, and detuning errors in the control pulse frequency may be implemented directly with the following keywords:

```python
"calibration_error": alpha,
"detuning_error: omega
```

In this case the ideal X-90 rotation will be implemented as the unitary $$U_{X90} = \exp\left[ -i (\frac{\pi}{2} + alpha) (\cos(\omega) X + \sin(\omega) Y ) \right]$$. If a `"U_error"` keyword is specified this additional coherent error will then be applied after, followed by any specified incoherent errors.

##### CX Gate

A coherent error model for a CX gate implemented via a cross-resonance interaction with a the control pulse amplitude calibration error, and a ZZ-interaction error may be implemented directly with the following keywords:

```python
"calibration_error": beta,
"zz_error": gamma
```

In this case the unitary for the CX gate is implemented as *U<sub>CX</sub> = U<sub>L</sub>\*U<sub>CR</sub>\*U<sub>R</sub>* where, *U<sub>CR</sub>* is the cross-resonance unitary, and *U<sub>L</sub>*, *U<sub>R</sub>* are the ideal local unitary rotations that would convert this to a CX in the ideal case. The ideal CR unitary is given by $$ U_{CR} = \exp\left[ -i \frac{\pi}{2} \frac{XZ}{2} \right]$$, where qubit-0 is the control, and qubit-1 is the target. The noisy CR gate with the above errors is given by
$$ U_{CR} = \exp\left[ -i (\frac{\pi}{2} + \beta) ( \frac{X \otimes Z}{2} + \gamma \frac{Z\otimes Z}{2}) \right]$$,

If a `"U_error"` keyword is specified this additional coherent error will then be applied after, followed by any specified incoherent errors.

### Thermal Relaxation Error

```python
"relaxation_rate": r
"thermal_populations": [p0, p1]
```

Specifies the parameters for the *T<sub>1</sub>* relaxation error of a system (with *T<sub>2</sub>=T<sub>1</sub>*). The probability of a relaxation error for a get of length *t* is given by $p_{err} = 1-exp(-t r) $. If a relaxation error occurs the system be reset to the 0 or 1 state with probability *p0* and *p1 = 1-p0* respectively.

Note that for single qubit gates the relaxation error occurs the noisy (or ideal) gate is not applied to the state.

### Reset Error

This error models the system being reset into an incorrect computational basis state. If used in combination with the `"reset"` gate error, the gate error is applied in addition *afterwards*.

```python
"reset_error": p
```

When a qubit is reset it be set to the 0 or 1 states with probabilities *1-p* and *p* respectively. This error is applied *before* the `"reset"` gate error is applied to the reset qubit.


### Measurement Readout Error

This error models incorrectly assigning the value of a classical bit after a measurement. It does not affect the quantum state of the system at all, only the classical registers. If used in combination with the `"measure"` gate error, the gate error is applied first, and then the readout error is applied to the measurement of the resulting quantum state.


```python
"readout_error": m
```

- If a system is measured to be in the 0 (or 1) state, the value recorded in the classical bit will be correctly recorded as 0 (or 1) with probability *1-m*, and incorrectly recorded as 1 (or 0) with probability *m*.

```python
"readout_error": [m0, m1]
```

- If a system is measured to be in the 0 state, the correct (0) and incorrect (1) outcome will be recorded with probability *1-m0* and *m0* respectively.
- If the system is measured to be in the 1 state the correct (1) and incorrect (0) outcome will be recorded with probability *1-m1* and *m1* respectively.

## Full Config Specification

An example of a configuration file for a 2-qubit circuit using all options is given below:

```python
"config": {
    "shots": 4,
    "seed": 0,
    "shots_threads": 4,[]()
    "data": [
        "classical_state",
        "quantum_state",
        "quantum_state_probs",
        "target_states_inner",
        "target_states_probs",
        "saved_quantum_states"
    ],
    "initial_state": [1, 0, 0, 1],
    "target_states": [
        [1, 0, 0, 1],
        [1, 0, 0, -1]
        [[1, 0], [0, 0], [0, 0], [0, 1]],
        [[1, 0], [0, 0], [0, 0], [0, -1]]
    ],
    "renom_target_states: True,
    "chop": 1e-10,
    "noise_params": {
        "reset_error": p_reset,
        "readout_error: [p_m0, p_m1],
        "relaxation_rate": r,
        "thermal_populations": [p0, p1],
        "measure": {
            "p_depol": p_meas,
            "p_pauli": [pI_meas, pX_meas, pY_meas, pZ_meas],
            "gate_time": t_meas,
            "U_error": matrix_meas
        },
        "measure": {
            "p_depol": p_res,
            "p_pauli": [pI_res, pX_res, pY_res, pZ_res],
            "gate_time": t_res,
            "U_error": matrix_res
        },
        "id": {
            "p_depol": p_id,
            "p_pauli": [pI_id, pX_id, pY_id, pZ_id],
            "gate_time": t_id,
            "U_error": matrix_id
        },
        "U": {
            "p_depol": p_u,
            "p_pauli": [pI_u, pX_u, pY_u, pZ_u],
            "gate_time": t_u,
            "U_error": matrix_u
        },
        "X90": {
            "p_depol": p_x90,
            "p_pauli": [pI_x90, pX_x90, pY_x90, pZ_x90],
            "gate_time": t_X90,
            "U_error": matrix_x90,
            "amp_error": alpha,
            "phase_error": omega
        },
        "CX": {
            "p_depol": p_cx,
            "p_pauli": [pII_cx, pIX_cx, pIY_cx, pIZ_cx,
                        pXI_cx, pXX_cx, pXY_cx, pXZ_cx,
                        pYI_cx, pYX_cx, pYY_cx, pYZ_cx,
                        pZI_cx, pZX_cx, pZY_cx, pZZ_cx],
            "gate_time": t_cx,
            "U_error": matrix_cx,
            "amp_error": alpha,
            "zz_error": gamma
    }
}
```

## Acknowledgements

The development and implementation of approximate noise models in this software was funded by the Intelligence Advanced Research Projects Activity (IARPA), via the Army Research Office contract W911NF-16-1-0114.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
