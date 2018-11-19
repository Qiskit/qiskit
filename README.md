# Qiskit Terra

[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/Qiskit/qiskit-terra.svg?branch=master)](https://travis-ci.org/Qiskit/qiskit-terra)
[![Build Status IBM Q](https://travis-matrix-badges.herokuapp.com/repos/Qiskit/qiskit-terra/branches/master/8)](https://travis-ci.org/Qiskit/qiskit-terra)

**Qiskit**  is a software development kit  for
developing quantum computing applications and working with NISQ (Noisy-Intermediate Scale Quantum) computers.

Qiskit is made up elements that each work together to enable quantum computing. This element is **Terra** 
and is the foundation on which the rest of Qiskit is built (see this [post](https://medium.com/qiskit/qiskit-and-its-fundamental-elements-bcd7ead80492) for an overview).


## Installation


We encourage installing Qiskit via the PIP tool (a python package manager):

```bash
pip install qiskit
```

PIP will handle all dependencies automatically for us and you will always install the latest (and well-tested) version.

At least [Python 3.5 or later](https://www.python.org/downloads/) is needed for using Qiskit. In
addition, [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) is recommended
for interacting with the tutorials.
For this reason we recommend installing the [Anaconda 3](https://www.continuum.io/downloads)
python distribution, as it comes with all of these dependencies pre-installed.

See [installing](doc/install.rst) Qiskit for detailed instructions, how to build from source and using environments. 


## Creating your first quantum program

Now that Qiskit is installed, it's time to begin working with Terra.

We are ready to try out a quantum circuit example, which is simulated locally using 
the Qiskt Aer element. This is a simple example that makes an entangled state.

```
$ python
```

```python
>>> from qiskit import *
>>> q = QuantumRegister(2)
>>> c = ClassicalRegister(2)
>>> qc = QuantumCircuit(q, c)
>>> qc.h(q[0])
>>> qc.cx(q[0], q[1])
>>> qc.measure(q, c)
>>> backend_sim = Aer.get_backend('qasm_simulator')
>>> result = execute(qc, backend_sim).result()
>>> print(result.get_counts(qc))
```

In this case, the output will be:

```python
{'counts': {'00': 513, '11': 511}}
```

A script is available [here](examples/python/hello_quantum.py), where we also show how to
run the same program on a real quantum computer via IBMQ.  

### Executing your code on a real quantum chip

You can also use Qiskit to execute your code on a
**real quantum chip**.
In order to do so, you need to configure Qiskit for using the credentials in
your IBM Q account:

#### Configure your IBMQ credentials

1. Create an _[IBM Q](https://quantumexperience.ng.bluemix.net) > Account_ if you haven't already done so.

2. Get an API token from the IBM Q website under _My Account > Advanced > API Token_. 

3. Take your token from step 2, here called `MY_API_TOKEN`, and run:

   ```python
   >>> from qiskit import IBMQ
   >>> IBMQ.save_account('MY_API_TOKEN')
    ```

4. If you have access to the IBM Q Network features, you also need to pass the
   url listed on your IBM Q account page to `save_account`.

After calling `IBMQ.save_account()`, your credentials will be stored on disk.
Once they are stored, at any point in the future you can load and use them
in your program simply via:

```python
>>> from qiskit import IBMQ
>>> IBMQ.load_accounts()
```

For those who do not want to save there credentials to disk please use

```python
>>> from qiskit import IBMQ
>>> IBMQ.enable_account('MY_API_TOKEN')
``` 

and the token will only be active for the session. For examples using Terra with real 
devices we have provided a set of examples in **examples/python** and we suggest starting with [using_qiskit_terra_level_0.py](examples/python/using_qiskit_terra_level_0.py) and working up in 
the levels.

## Contribution guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](.github/CONTRIBUTING.rst). This project adheres to Qiskit's [code of conduct](.github/CODE_OF_CONDUCT.rst). By participating, you are expect to uphold to this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-terra/issues) for tracking requests and bugs. 
Please use our [slack](https://qiskit.slack.com) for discussion. To join our Slack community use the [link](https://join.slack.com/t/qiskit/shared_invite/enQtNDc2NjUzMjE4Mzc0LTMwZmE0YTM4ZThiNGJmODkzN2Y2NTNlMDIwYWNjYzA2ZmM1YTRlZGQ3OGM0NjcwMjZkZGE0MTA4MGQ1ZTVmYzk). To ask questions to [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).



### Next Steps

Now you're set up and ready to check out some of the other examples from our
[Qiskit Tutorial](https://github.com/Qiskit/qiskit-tutorial) repository.


## Authors

Qiskit Terra is the work of [many people](https://github.com/Qiskit/qiskit-terra/graphs/contributors) who contribute
to the project at different levels.

## License

[Apache License 2.0](LICENSE.txt)