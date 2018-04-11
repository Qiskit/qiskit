# Quantum Information Software Kit (QISKit)

[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/QISKit/qiskit-sdk-py.svg?branch=master)](https://travis-ci.org/QISKit/qiskit-sdk-py)

**QISKit**是一个用于和 [OpenQASM](https://github.com/QISKit/qiskit-openqasm)和
[IBM Q experience (QX)](https://quantumexperience.ng.bluemix.net/)协同工作的软件开发工具包(SDK)。

利用 **QISKit** 来编写量子计算程序、编译、并且在一些终端上运行（实时在线量子处理器，在线模拟器，和本地模拟器）。 在在线后端， QISKit 使用我们的 [python API client](https://github.com/QISKit/qiskit-api-py)
与 IBM Q experience建立连接。

**我们使用 GitHub issues 来追踪需求和错误。详见**
[IBM Q experience community](https://quantumexperience.ng.bluemix.net/qx/community) **中的提问和讨论。**
**如果您有意对 QISKit 做出贡献，请参见我们的**
[contribution guidelines](https://github.com/QISKit/qiskit-sdk-py/blob/master/CONTRIBUTING.rst)。

链接索引:

* [安装](#安装)
* [创建您的第一个量子程序](#创建您的第一个量子程序)
* [更多信息](#更多信息)
* [作者](#作者-按字母顺序)
* [版权许可证](#版权许可证)

## 安装

### 安装环境

安装使用 QISKit 需要 [Python 3.5](https://www.python.org/downloads/) 版本及以上。 并且建议使用[Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) 来参阅我们的教程。
因此我们推荐安装 [Anaconda 3](https://www.continuum.io/downloads)
开源Python 发行版本， 它包含了所有需要的安装包。
此外，如果有一些对量子信息的基本了解，在使用
QISKit的时候会很有帮助。如果您是量子领域的新手，请参见我们的
[User Guides](https://github.com/QISKit/ibmqx-user-guides)！

### 安装

我们鼓励您使用PIP 工具 (Python包管理工具)来安装 QISKit：
```
    pip install qiskit
```
PIP 会自动处理所有的依赖文件，并替您将它们更新到最新的（经过测试的）软件版本。
PIP 为以下平台预装有二进制版本：

* Linux x86_64
* Darwin
* Win64

如果您的运行平台不在以上的列表中，PIP 将会在安装的时候从安装源创建。这将需要预装 CMake 3.5 或更高的版本和至少一个由 [CMake](https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html)支持的开发环境。
如果在安装过程中 PIP 没能成功创建，不要担心，您最终还是会成功安装 QISKit，只是可能不能使用一些高阶的功能。无论如何，我们还会提供一个不那么快捷的python备选方案。


#### 配置您的安装环境

我们建议采用python虚拟环境来提升运行体验。更多信息请参见
[Environment Setup documentation](https://github.com/QISKit/qiskit-sdk-py/blob/master/doc/install.rst#3.1-Setup-the-environment) 。

## 创建您的第一个量子程序

当SDK安装好了之后，我们可以开始使用QISKit了。

我们可以试验一个量子电路的例子，它将在本地模拟器上运行。

这是一个产生叠加态的简单例子：

```python
# Import the QISKit SDK
import qiskit

# Create a Quantum Register called "qr" with 2 qubits
qr = qiskit.QuantumRegister("qr", 2)
# Create a Classical Register called "cr" with 2 bits
cr = qiskit.ClassicalRegister("cr", 2)
# Create a Quantum Circuit called involving "qr" and "cr"
qc = qiskit.QuantumCircuit(qr, cr)

# Add a H gate on the 0th qubit in "qr", putting this qubit in superposition.
qc.h(qr[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qc.cx(qr[0], qr[1])
# Add a Measure gate to see the state.
# (Omitting the index applies an operation on all qubits of the register(s))
qc.measure(qr, cr)

# Create a Quantum Program for execution 
qp = qiskit.QuantumProgram()
# Add the circuit you created to it, and call it the "bell" circuit.
# (You can add multiple circuits to the same program, for batch execution)
qp.add_circuit("bell", qc)

# See a list of available local simulators
print("Local backends: ", qiskit.backends.discover_local_backends())

# Compile and run the Quantum Program on a simulator backend
sim_result = qp.execute("bell", backend='local_qasm_simulator', shots=1024, seed=1)

# Show the results
print("simulation: ", sim_result)
print(sim_result.get_counts("bell"))
```

在这个例子中，输出是（大约是由于随机波动）：
```
COMPLETED
{'counts': {'00': 512, '11': 512}}
```
可以在 [这里](https://github.com/QISKit/qiskit-sdk-py/blob/master/examples/python/hello_quantum.py)找到此例的脚本。

### 在一个真实的量子芯片上执行您的程序

您也可以使用 QISKit 在一个
[真实的量子芯片](https://github.com/QISKit/ibmqx-backend-information)上执行您的程序。
为此，您需要配置 SDK 来为您的 Quantum Experience Account使用证书：

#### 配置您的 API 令牌和 QE 证书

1. 建立一个 [IBM Q experience](https://quantumexperience.ng.bluemix.net)
   账号，如果您还没有的话。
2. 在IBM Q experience网页上取得一个API 令牌： "`My Account`" >
   "`Personal Access Token`"。这个API令牌将使您可以在IBM Q 体验后端上运行您的程序。
   [示例](https://github.com/QISKit/qiskit-sdk-py/blob/master/doc/example_real_backend.rst)。
3. 之后我们将创建一个新的文件叫 `Qconfig.py` 并在其中插入 API 令牌。此文件必须包含以下内容：
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
4. 用您在第2步中获得的 API 令牌替换 `'MY_API_TOKEN'` 。
5. 如果您有IBM Q 的相关访问权限，您仍需为您的hub、group、和project设置数值。可以通过填 `config` 的变量来将其设置为您 IBM Q 账户页面上的数值。

当 `Qconfig.py` 文件设置好了之后，您需要将其移动到与您的程序/教程在同一目录/文件夹下，这样它可以被调用使 `QuantumProgram.set_api()` 函数生效。例如：
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

更多详细信息请参见我们的
[QISKit documentation](https://www.qiskit.org/documentation/)。

### 下一步

现在您已经设置好并准备好查看我们的
[教程](https://github.com/QISKit/qiskit-tutorial) 库中的更多实例了。 首先选择
[index tutorial](https://github.com/QISKit/qiskit-tutorial/blob/master/index.ipynb) 然后选择 [‘Getting Started’ example](https://github.com/QISKit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)。
如果您已经有安装有 [Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html)，
那么您可以复制和修改这些notebooks来创建您自己的实验。

如要将教程安装在 QISKit SDK 中，请参见
[安装详述](https://github.com/QISKit/qiskit-sdk-py/blob/master/doc/install.rst#Install-Jupyter-based-tutorials)。 完整的 SDK
文档请参见 [*doc* directory](https://github.com/QISKit/qiskit-sdk-py/blob/master/doc/qiskit.rst) 和
[QISKit 官网](https://www.qiskit.org/documentation)。

## 更多信息

欲知更多有关如何使用 QISKit、指导实例、和其他有用链接的信息，请见以下资源：

* **[用户指南](https://github.com/QISKit/ibmqx-user-guides)**，
  一个学习量子信息和计算的很好的入门资料
* **[教程](https://github.com/QISKit/qiskit-tutorial)**，
  例如要学习notebooks，可以参见 [index](https://github.com/QISKit/qiskit-tutorial/blob/master/index.ipynb) 和 [‘Getting Started’ Jupyter notebook](https://github.com/QISKit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)
* **[OpenQASM](https://github.com/QISKit/openqasm)**，
  可以找到更多有关QASM代码的信息和实例
* **[IBM Quantum Experience Composer](https://quantumexperience.ng.bluemix.net/qx/editor)**，
  一个与真实和模拟量子计算机交互的GUI
* **[QISkit Python API](https://github.com/QISKit/qiskit-api-py)**，一个在Python中使用IBM Quantum Experience 的API


QISKit 最早是由[IBM Research](http://www.research.ibm.com/)研究中心的
[IBM-Q](http://www.research.ibm.com/ibm-q/) 团队的研究人员和开发人员开发的，
旨在提供一个与量子计算机配套的高水平的开发工具包。

欲知更多有关 QISKit 和更广泛地有关量子计算的提问和讨论请访问 [IBM Q experience community](https://quantumexperience.ng.bluemix.net/qx/community)。 如果您有兴趣为 QISKit 做出贡献，请参见我们的 [contribution guidelines](https://github.com/QISKit/qiskit-sdk-py/blob/master/CONTRIBUTING.rst)。

## 多语言指导

* **[Korean Translation](https://github.com/QISKit/qiskit-sdk-py/blob/master/doc/ko/README.md)**， 基本的韩语指导。

## 作者 (按字母顺序)

QISKit was originally authored by
Luciano Bello, Jim Challenger, Andrew Cross, Ismael Faro, Jay Gambetta, Juan Gomez,
Ali Javadi-Abhari, Paco Martin, Diego Moreda, Jesus Perez, Erick Winston and Chris Wood.

And continues to grow with the help and work of [many people](https://github.com/QISKit/qiskit-sdk-py/tree/master/CONTRIBUTORS.md) who contribute
to the project at different levels.

## 版权许可证

此项目使用了 [Apache License Version 2.0 software license](https://www.apache.org/licenses/LICENSE-2.0)。
