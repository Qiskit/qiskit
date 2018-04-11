# Quantum Information Software Kit (QISKit)

[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/QISKit/qiskit-sdk-py.svg?branch=master)](https://travis-ci.org/QISKit/qiskit-sdk-py)

**QISKit**은 [OpenQASM](https://github.com/QISKit/qiskit-openqasm)과 [IBM Q experience (QX)](https://quantumexperience.ng.bluemix.net/)에서 사용할 수 있는 SDK(software development kit)입니다. 

양자컴퓨팅 프로그램을 만들고, 컴파일하고, 실행하기 위해 **QISKit**을 사용하세요. 실행에 있어 QISKit은 온라인상 접근 가능한 실제 양자 프로세서, 온라인 시뮬레이터, 로컬 시뮬레이터의 백엔드 환경을 지원합니다. 온라인 백엔드 환경에서, QISKit은 IBM Q experience와 연결하기 위해 [python API client](https://github.com/QISKit/qiskit-api-py)를 사용합니다.

우리는 **GitHub issue**를 요청과 버그 추적에 사용합니다. 질문과 토론을 위해[IBM Q experience community](https://quantumexperience.ng.bluemix.net/qx/community)를 살펴보세요.  만일 여러분이 QISKit개발에 기여하기 원한다면 우리의 **[contribution guidelines](CONTRIBUTING.rst)** 을 살펴보세요.

목차 링크:

* [설치](#installation)
* [첫 번째 양자컴퓨팅 프로그램 만들기](#creating-your-first-quantum-program)
* [더 많은 정보](#more-information)
* [저자들](#authors-alphabetical)
* [라이센스](#license)

## Installation

### Dependencies

적어도 [파이썬 Python 3.5 혹은 그 이상의 버전이](https://www.python.org/downloads/) QISKit을 사용하기 위해 필요합니다. 그리고, [Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html)을 튜토리얼의 실행을 위해 권장하는 바입니다. 이러한 연유로 우리는 [Anaconda 3](https://www.continuum.io/downloads)를 사용하는 걸 권장합니다. 아나콘다에는 대부분의 필요한 툴이 미리 설치되어 있습니다. 또한, 양자 정보에 대한 일반적인 이해는 QISKit과 상호작용하는 데에 매우 도움이 됩니다. 만일 여러분이 양자컴퓨팅 분야에 새롭게 배운다면 다음의 [사용자 가이드](https://github.com/QISKit/ibmqx-user-guides)를 살펴보세요.

### PIP Installation

파이썬이 더 익숙한 사람들에게, QISKit을 설치하는 가장 빠른 방법은 PIP 툴(파이썬 패키지 매니저)을 이용하는 방식입니다.

```
    pip install qiskit
```

### Source Installation

QISKit SDK 레파지토리를 여러분의 로컬 머신에 Clone하는 다른 방법으로는 클론한 디렉토리를 바꾸는 방법이 있습니다. 

#### Manual download

수동 다운로드 방법으로 이 웹페이지의 상단의 "Clone or download" 버튼을 누르세요 (혹은 git clone 커맨드 상에 보이는 URL을 통해), 만일 필요하다면 압축을 풀고 폴더 이름을 터미널 상에서 다음과 같이 바꾸세요. **qiskit-sdk-py folder** 

#### Git download

혹은, 만일 여러분이 이미 Git을 설치했다면, 다음의 커맨드를 실행하세요:
```
    git clone https://github.com/QISKit/qiskit-sdk-py
    cd qiskit-sdk-py
```

#### Setup your enviroment

우리는 여러분의 사용자 경험을 향상시키기 위해 파이썬의 가상 환경을 사용하는 걸 추천합니다. 우리의 [Environment Setup documentation](doc/install.rst#3.1-Setup-the-environment) 을 더 많은 정보를 얻기위해 참조하세요.

## Creating your first Quantum Program

SDK 설치가 끝났습니다. 이제 QISKit으로 작업을 해볼 차례입니다. 우리는 이미 QASM예제를 로컬 시뮬레이터 상에서 실행할 준비를 마쳤습니다. 이것은 간단한 중첩 예제 입니다. 

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

이 경우에 결과물은 다음과 같을 것입니다(불규칙 변동으로 인해):

```
COMPLETED
{'00': 509, '11': 515}
```
여러분은 또한 QISKit을 사용하여 여러분의 코드를 [real Quantum Chip](https://github.com/QISKit/ibmqx-backend-information)에서 실행시킬 수 있습니다.

 먼저, API 토큰을 얻으세요:
 
-  [IBM Q experience](https://quantumexperience.ng.bluemix.net) 계정을 생성하세요. 
-  IBM Q experience 웹 페이지에서 API token을 얻으세요. 토큰은 “My Account” > “Personal Access Token”에 위치해 있습니다. 

이 API 토큰은 여러분의 프로그램을 IBM Q experience 백엔드 환경에서 실행할 수 있도록 해줄 것입니다. [예제](doc/example_real_backend.rst).

이에 대해 더 많은 정보를 알고 싶다면, 다음의 링크를 참조하세요. [our QISKit documentation](doc/qiskit.rst).


### Next Steps

이제 여러분은 다른 [튜토리얼](https://github.com/QISKit/qiskit-tutorial)들까지도 실행할 수 있는 준비와 환경 구성을 마쳤습니다. [index tutorial](https://github.com/QISKit/qiskit-tutorial/blob/master/index.ipynb)을 시작하세요. 그리고 [‘Getting Started’ 예제](https://github.com/QISKit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)로 이동하세요. 만일 여러분이 이미 [Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html)을 설치했다면, 여러분은 해당 노트북을 복사하고 수정하여 여러분 만의 노트북을 만들 수 있습니다. 

튜토리얼을 QISKit의 일부로서 설치하기 위해 다음의 링크를 확인하세요. [installation details](doc/install.rst#Install-Jupyter-based-tutorials). 완전한 SDK문서는 다음의 링크에서 확인할 수 있습니다. [*doc* directory](doc/qiskit.rst).

## More Information

다음은 QISKit을 어떻게 써야하는지에 대한 더많은 정보와 튜토리얼 예제, 그리고 몇몇의 도움이 될만한 링크들 입니다. 한 번 살펴보세요. 
* **[User Guides](https://github.com/QISKit/ibmqx-user-guides)**,
  양자정보와 양자컴퓨팅에 대해 배울 수 있는 좋은 시작점 
* **[Tutorials](https://github.com/QISKit/qiskit-tutorial)**,
  예를 들어 Jupyter 노트북의 경우 [index](https://github.com/QISKit/qiskit-tutorial/blob/master/index.ipynb)를 참고하세요. 그리고 [‘Getting Started’ Jupyter notebook](https://github.com/QISKit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)도 함께 보세요. 
* **[OpenQASM](https://github.com/QISKit/openqasm)**,
  QASM에 대한 추가적인 정보와 예제
* **[IBM Quantum Experience Composer](https://quantumexperience.ng.bluemix.net/qx/editor)**,
  실제 양자컴퓨터 및 시뮬레이션된 양자컴퓨터와의 GUI 인터페이스를 통한 인터렉션
* **[QISkit Python API](https://github.com/QISKit/qiskit-api-py)**, 파이썬을 통해 IBM Quantum Experience를 사용할 수 있는 API 

QISKit은 본래 [IBM Research](http://www.research.ibm.com/)연구팀과 [IBM-Q](http://www.research.ibm.com/ibm-q/)개발팀에 의해 양자컴퓨터에 대한 고수준(high level) 개발킷을 제공할 목적으로 개발되었습니다. 
질의응답과 토론 그리고 양자컴퓨팅에 대해 더 넓게 살펴보기 위해 [IBM Q experience community](https://quantumexperience.ng.bluemix.net/qx/community)를 방문하세요. 만일 여러분이 QISKit에 기여하길 원한다면 다음의 가이드라인을 살펴보세요. [contribution guidelines](CONTRIBUTING.rst).

## Multilanguage guide

* **[Korean Translation](https://github.com/QISKit/qiskit-sdk-py/tree/master/doc/ko/README-ko.md)**, 한글 기본 가이드 라인

## Authors (alphabetical)

QISKit was originally authored by
Luciano Bello, Jim Challenger, Andrew Cross, Ismael Faro, Jay Gambetta, Juan Gomez,
Ali Javadi-Abhari, Paco Martin, Diego Moreda, Jesus Perez, Erick Winston and Chris Wood.

And continues to grow with the help and work of [many people](https://github.com/QISKit/qiskit-sdk-py/tree/master/CONTRIBUTORS.md) who contribute
to the project at different levels.

## License

본 프로젝트는 다음의 라이센스가 적용됩니다. [Apache License Version 2.0 software license](https://www.apache.org/licenses/LICENSE-2.0).


