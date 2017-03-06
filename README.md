# Python SDK

Python software development kit (SDK) and Jupyter notebooks for working with OPENQASM and the IBM Quantum Experience (QX).

Related external projects:

- [Python API](https://github.com/IBM/qiskit-api-py)
- [OPENQASM](https://github.com/IBM/qiskit-openqasm)

## Organization

The *scripts* directory contains Jupyter notebooks showing how to use the [Python API](https://github.com/IBM/qiskit-api-py) with [OPENQASM](https://github.com/IBM/qiskit-openqasm).

## Setup Python Virtual Enviroment
### Anaconda
To create a new Virtual Enviroment:
> conda create -n QISKitenv python=3 pip

use it:

> source activate QISKitenv

### Install Dependencies
> pip install -r requires.txt


### Dependencies problems.

When you try to install the dependencies "pip install --upgrade IBMQuantumExperience", if you have the next error:

* Cannot remove entries from nonexistent file [PATH]/easy-install.pth

You can fix it using:

> curl https://bootstrap.pypa.io/ez_setup.py -o - | python
