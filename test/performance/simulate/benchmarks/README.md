# QCS Perf Benchmark

## OVERVIEW

This benchmark is a quantum-software bencmark suite that helps evaluate quantum volumes and performance of quntum computers and simulators registered as backends in QISKit API. 

## Getting Started

### Setup QISKit

see https://github.com/QISKit/qiskit-sdk-py/#installation

### Run benchmarks

You can run benchmark `qcs_perf.py` with following options.

* `-a`: specify an appplication to be evaluated (qft, qv, bv, cc)
* `-b`: specify a backend supportted by QISKit or ProjectQ and Qsharp
* `-s`: specify a qubit number to start evaluation
* `-e`: specify a qubit number to end evaluation
* `-d`: specify a depth to be evaluated (optional)
* `-v`: verify simulation results (optional)
* `-l`: show the list of benchmark scenario (optional)
* `-sd`: set the initial seed for the simulator
* `-as`: set the initial seed for the application

For example, the following commands run qft from 10 to 20 qubit with local_qasm_simulator.
```
$ python3 qcs_perf.py -a qft -b local_qasm_simulator -s 10 -e 20
$ python3 qcs_perf.py -a qft -b local_qasm_simulator -s 10 -e 20
``` 

If you install Q# or ProjectQ on your host, you can run the benchmark on them.
```
$ python3 qcs_perf.py -a qft -b Qsharp -s 10 -e 20
$ python3 qcs_perf.py -a qft -b ProjectQ -s 10 -e 20
``` 

## Applications

### Fourier Transform (qft)

https://github.com/QISKit/qiskit-tutorial/blob/master/appendix/more_qis/fourier_transform.ipynb

### Quantum Volume (qv)
Generate randomized circuits for Quantum Volume analysis.

### Bernstein-Vazirani algorithm (bv)
This program is based on the Bernstein-Vazirani algorithm in the [QISKit-tutorial](https://github.com/QISKit/qiskit-tutorial).

### Counterfeit-Coin Finding algorithm (cc)
This program is based on the Counterfeit-Coin Finding algorithm in the [QISKit-tutorial](https://github.com/QISKit/qiskit-tutorial).

