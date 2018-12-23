# Quantum Information Science Kit (Qiskit)

[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/Qiskit/qiskit-terra.svg?branch=master)](https://travis-ci.org/Qiskit/qiskit-terra)
[![Build Status IBM Q](https://travis-matrix-badges.herokuapp.com/repos/Qiskit/qiskit-terra/branches/master/8)](https://travis-ci.org/Qiskit/qiskit-terra)

量子信息科学套件（缩写 **Qiskit**）是一个用于 NISQ（嘈杂中型量子）计算机，例如 [IBM Q](https://quantumexperience.ng.bluemix.net/) ，开发量子计算应用的软件开发工具包（SDK）。

**Qiskit** 包含了多个帮助量子计算的使用的组件。此仓库的组件是 **Terra**，是 **Qiskit** 其余部分的基础（详见[此帖](https://medium.com/qiskit/qiskit-and-its-fundamental-elements-bcd7ead80492)以了解组件间的关系（英文））。

**我们使用 GitHub Issues 来追踪需求和错误。详见**
[slack](https://qiskit.slack.com) **进行提问和讨论。**

利用 **Qiskit** 来编写量子计算程序、编译、并且在一些终端上运行（实时在线量子处理器，在线模拟器，和本地模拟器）。 在在线后端， Qiskit 使用我们的 [python API client](https://github.com/Qiskit/qiskit-api-py)
与 IBM Q experience建立连接。

**如果您有意对 Qiskit 做出贡献，请参见我们的**
[贡献指南](https://github.com/Qiskit/qiskit-terra/blob/master/.github/CONTRIBUTING.rst)。

链接索引:

* [安装](#安装)
* [创建您的第一个量子程序](#创建您的第一个量子程序)
* [更多信息](#更多信息)
* [作者](#作者-按字母顺序)

## 安装

### 安装环境

安装使用 Qiskit 需要 [Python 3.5](https://www.python.org/downloads/) 及更新的版本。我们建议在参考教程的过程中使用 [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) 。我们也推荐安装 [Anaconda 3](https://www.continuum.io/downloads)
Python 发行版本，因为它内置了所有需要的安装包。此外，如果有一些对量子信息的基本了解，在使用 Qiskit 的时候会很有帮助。

此外，对量子信息的基本理解对使用 Qiskit 相当有帮助。如果您刚刚开始接触量子，可以从
[IBM Q Experience（英文）](https://quantumexperience.ng.bluemix.net)开始！

### 安装

我们鼓励使用 PIP 工具（Python包管理工具）来安装 Qiskit：

```bash
pip install qiskit
```
PIP 会自动处理所有的依赖文件，并始终将它们更新到最新的（经过测试的）版本。

PIP 为以下平台预装有二进制版本：

* Linux x86_64
* Darwin
* Win64

如果您的运行平台不在以上的列表中，PIP 将会在安装时编译。这将需要预装 CMake 3.5 或更高的版本和至少一个由 [CMake](https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html)支持的开发环境。

如果在安装过程中 PIP 未能成功编译。不要担心，Qiskit 仍能安装成功，这只会使部分高性能组件不能使用。无论如何，仍会回滚至一个较慢的、由 Python 实现的备用方案。

#### 配置您的安装环境

我们建议采用 Python 虚拟环境来提升您的使用体验。更多信息请参见[环境配置指南（英文）](https://github.com/Qiskit/qiskit-terra/blob/master/doc/install.rst#3.1-Setup-the-environment) 。

## 创建您的第一个量子程序

当 Qiskit 安装好了之后，我们可以开始使用 Terra 了。

我们将试验一个量子电路的例子，并在本地模拟器上运行。

这是一个产生叠加态的简单例子：

```python
# 导入 SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer
# 创建 2 个量子位的寄存器
q = QuantumRegister(2)
# 创建 2 个经典寄存器
c = ClassicalRegister(2)
# 创建量子电路
qc = QuantumCircuit(q, c)
# 在量子位 0 上创建 H 量子门，使其进入叠加态
qc.h(q[0])
# 在量子位 0 上创建 CX (CNOT) 量子门，
# 设置目标为量子位 1，使其进入贝尔态
qc.cx(q[0], q[1])
# 创建观测门以观测状态
qc.measure(q, c)
# 查看所有可用的后端
print("Aer backends: ", Aer.backends())
# 编译并使用模拟后端运行
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend_sim)
result_sim = job_sim.result()
# 查看结果
print("simulation: ", result_sim )
print(result_sim.get_counts(qc))
```
在这个情况下，结果应该为：

```python
COMPLETED
{'counts': {'00': 512, '11': 512}}
```

可以在[这里](examples/python/hello_quantum.py)查看到我们将此例子通过 IBMQ 运行在真实量子计算机的脚本。

### 在真实量子芯片上执行您的程序

你也可以将这段代码运行到一个 **真实的量子计算机** 上。您需要配置 Qiskit 使用您 IBM Q 账户的证书。

#### 配置您的 API 令牌和 QX 证书

1. 如果您还没有 IBM Q 帐号的话，请建立一个 [IBM Q](https://quantumexperience.ng.bluemix.net) 帐号。
2. 在 IBM Q 网站 _My Account > Advanced > API Token_ 中获得一个 API 令牌。API 令牌使你可以 IBM Q 后端来执行你的程序。
3. 我们现在将第二步中获取的令牌添加到 Qiskit 中。请调用 `IBMQ.save_account()` 函数来导入：
  ```python
  from qiskit import IBMQ
  IBMQ.save_account('MY_API_TOKEN')
  ```

4. 如果你有使用 IBM Q 网络特性，您还需要将 IBM Q 账户的链接传递到 `save_account` 函数中。调用后，您的证书会被保存到本地硬盘上。一旦保存，您可以简单通过调用来使用证书：

```python
from qiskit import IBMQ
IBMQ.load_accounts()
```

如果您不想保存证书到硬盘，请使用如下代码：

```python
from qiskit import IBMQ
IBMQ.enable_account('MY_API_TOKEN')
``` 

这会让令牌只在当前上下文中有效。对于在真实设备上使用 Terra，我们在 **examples/python** 中提供了许多例子。我们推荐您从 [using_qiskit_terra_level_0.py](examples/python/using_qiskit_terra_level_0.py) 开始。

对于更多安装 Qiskit 的细节以及使用 IBM Q 证书的备用方法（例如通过环境变量，以及早期版本中 `Qconfig.py` 支持），请查看 [Qiskit 文档](https://www.qiskit.org/documentation/)。

### 下一步

现在您已经配置好环境，可以查看[教程](https://github.com/Qiskit/qiskit-tutorial)库中的更多实例了。首先请先看
[index tutorial](https://github.com/Qiskit/qiskit-tutorial/blob/master/index.ipynb) ，然后查看 [‘Getting Started’ example](https://github.com/Qiskit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)。

如果您已经有安装有 [Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html)，
那么您可以复制和修改这些 notebooks 来创建您自己的实验。

如要将教程安装在 Qiskit SDK 中，请参见
[安装详述](https://github.com/Qiskit/qiskit-terra/blob/master/doc/install.rst#Install-Jupyter-based-tutorials)。完整的 SDK 文档请参见 [*doc* directory](https://github.com/Qiskit/qiskit-terra/blob/master/doc/qiskit.rst) 和
[Qiskit 官网](https://www.qiskit.org/documentation)。

## 更多信息

欲知更多有关如何使用 Qiskit、指导实例、和其他有用链接的信息，请见以下资源：

* **[Qiskit Aqua](https://github.com/Qiskit/aqua)**，
  提供了许多量子算法的组件
* **[教程](https://github.com/Qiskit/qiskit-tutorial)**，
  含有大量 notebook 例子，可以参见 [index](https://github.com/Qiskit/qiskit-tutorial/blob/master/index.ipynb) 和 [‘Getting Started’ Jupyter notebook](https://github.com/Qiskit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)
* **[OpenQASM](https://github.com/Qiskit/openqasm)**，
  可以找到更多有关 QASM 代码的信息和实例
* **[IBM Q Experience](https://quantumexperience.ng.bluemix.net)**，
  提供了一个操作真实量子计算机的图形介面

## 多语言指导

* **[Korean Translation](https://github.com/Qiskit/qiskit-terra/blob/master/doc/ko/README.md)**，基本的韩语指南。

## 作者 (按字母顺序)

Qiskit was originally authored by
Luciano Bello, Jim Challenger, Andrew Cross, Ismael Faro, Jay Gambetta, Juan Gomez,
Ali Javadi-Abhari, Paco Martin, Diego Moreda, Jesus Perez, Erick Winston and Chris Wood.

And continues to grow with the help and work of [many people](https://github.com/Qiskit/qiskit-terra/graphs/contributors) who contribute
to the project at different levels.
