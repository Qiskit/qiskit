# Quantum Information Science Kit (Qiskit)

[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/Qiskit/qiskit-terra.svg?branch=master)](https://travis-ci.org/Qiskit/qiskit-terra)

양자정보과학 키트(The Quantum Information Science Kit), 줄여서 **Qiskit**(키스킷)은 양자정보용 응용 프로그램을 개발하기 위해 만들어진 소프트웨어 개발용 키트 (SDK)입니다. **Qiskit**은 노이즈가 많은 중간 크기의 양자컴퓨터(NISQ -- Noisy-Intermediate Scale Quantum --computers)와 같이 쓰일 수 있습니다.

Qiskit은 양자컴퓨터를 구동하게 하는 여러가지의 요소로 이루어져 있습니다. 이 폴더는 Terra(땅) - Qiskit이 세워질 수 있게 하는 기반을 제공하는 요소입니다.

버그와 그 밖에 요청사항이 있으시면 GitHub issue코너를 이용하세요. 질문과 토론은 Qiskit [slack](https://qiskit.slack.com)채널을 이용하시면 됩니다. Qiskit Slack채널에 가입하시려면 [링크](https://join.slack.com/t/qiskit/shared_invite/enQtNDc2NjUzMjE4Mzc0LTMwZmE0YTM4ZThiNGJmODkzN2Y2NTNlMDIwYWNjYzA2ZmM1YTRlZGQ3OGM0NjcwMjZkZGE0MTA4MGQ1ZTVmYzk)를 클릭하세요.

바로가기: 

* [Qiskit의 설치방법](#installation)
* [첫 번째 양자컴퓨팅 프로그램 만들기](#creating-your-first-quantum-program)
* [더 많은 정보](#more-information)
* [저자들](#authors-alphabetical)
* [라이센스](#license)

## Qiskit의 설치방법

### Qiskit 설치와 사용을 위해 필요하거나 도음이 되는 프로그램들

적어도 [파이썬 Python 3.5 이상의 버전이](https://www.python.org/downloads/) 필요합니다. 그리고, 튜토리얼의 실행을 위해 [Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html)의 사용을 추천합니다. 이러한 연유로 Qiskit을 쓰실 때는 [Anaconda 3](https://www.continuum.io/downloads)의 사용을 권장합니다. 아나콘다에는 대부분의 필요한 툴이 미리 설치되어 있습니다. 또한, 양자 정보에 대한 일반적인 이해는 Qiskit과 상호작용하는 데에 매우 도움이 됩니다. 만일 여러분이 양자컴퓨팅 분야에 새롭게 배운다면 다음의 [사용자 가이드](https://github.com/Qiskit/ibmqx-user-guides)를 살펴보세요.

### PIP 설치

파이썬이 더 익숙한 사람들에게, Qiskit을 설치하는 가장 빠른 방법은 PIP 툴(파이썬 패키지 매니저)을 이용하는 방식입니다.

```
    pip install qiskit
```

PIP으로 Qiskit을 설치하시면 모든 필수 프로그램들이 같이 자동으로 설치가 되기 때문에, 항상 검증된 최신 버전의 Qiskit을 쓰게 되실 수 있습니다. 

PIP 패키지는 아래의 플랫폼들을 위해 미리 만들어진 바이너리를 포함하고 있습니다.   

    Linux x86_64
    Darwin
    Win64

위 목록에 포함되지 않은 플랫폼을 사용하고 있으시다면, PIP은 설치시에 소스코드로 부터 바이너리를 생성하려고 할 것입니다. 바이너리의 생성에는 CMake 버전 3.5 이상의 버전과, CMake가 지원하는 최소한 하나의 바이너리 [빌드 환경](https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html)이 필요합니다.  

### 상급자용 설치 방법과 환경 설정 

더 편리한 사용을 위해서 파이썬의 가상환경의 사용을 권장합니다. 자세한 내용은 [설치방법](doc/install.rst)을 참고하세요.

## 첫번째 양자 프로그램 만들기

Qiskit의 설치가 끝났습니다. 이제 Terra를 써볼 차례입니다. We are ready to try out a quantum circuit example, which is simulated locally using the Qiskt Aer element.

This is a simple example that makes an entangled state. 우리는 Qiskit Aer(에어)에서 시뮬레이션 한 이미 양자회로 예제를 실행할 준비가 되었습니다.   마쳤습니다. 다음은 양자얽힘 상태를 만드는 간단한 예제입니다.  


```python
# Import the Qiskit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer

# Create a Quantum Register with 2 qubits.
q = QuantumRegister(2)
# Create a Classical Register with 2 bits.
c = ClassicalRegister(2)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)

# Add a H gate on qubit 0, putting this qubit in superposition.
qc.h(q[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qc.cx(q[0], q[1])
# Add a Measure gate to see the state.
qc.measure(q, c)

# See a list of available local simulators
print("Aer backends: ", Aer.backends())

# Compile and run the Quantum circuit on a simulator backend
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend_sim)
result_sim = job_sim.result()

# Show the results
print("simulation: ", result_sim )
print(result_sim.get_counts(qc))
```

이 경우에 결과물은 다음과 같을 것입니다:

```python
COMPLETED
{'counts': {'00': 512, '11': 512}}
```
이 예제는 [이곳](examples/python/hello_quantum.py)에서 예제 프로그램을 실제 IBMQ 양자컴퓨터에서 실행하는 방법과 함께 찾아보실 수 있습니다.,


#### IBMQ 계정과 보안증명

1. 먼저 _[IBM Q](https://quantumexperience.ng.bluemix.net) > 계정_ 을 만듭니다.  

2. IBM Q 홈페이지에 로그 인 하신 후에 API 토큰을 받습니다. 토큰은 _My Account > Advanced > API Token_ 애서 찾아보실 수 있습니다. 이 API token은 IBM Q 의 양자컴퓨터들을 사용하실 때 필요합니다. 

3. Qiskit 프로그램을 IBM Q 백엔드에서 실행하실 때, 보안 증명을 위해 2번 단계에서 생성된 토큰을 `IBMQ.save_account()` 함수에 넘깁니다. 예를들어, 토큰을 `MY_API_TOKEN`이라는 변수에 저장했다면:

   ```python
   from qiskit import IBMQ

   IBMQ.save_account('MY_API_TOKEN')
    ``` your credentials will be stored on disk.
    
Once they are stored, at any point in the future you can load and use them
in your program simply via:

```python
from qiskit import IBMQ

IBMQ.load_accounts()
```

For those who do not want to save there credentials to disk please use

```python
from qiskit import IBMQ

IBMQ.enable_account('MY_API_TOKEN')
``` 

and the token will only be active for the session. For examples using Terra with real 
devices we have provided a set of examples in **examples/python** and we suggest starting with [using_qiskit_terra_level_0.py](examples/python/using_qiskit_terra_level_0.py) and working up in 
the levels.
### Executing your code on a real quantum chip

You can also use Qiskit to execute your code on a
**real quantum chip**.
In order to do so, you need to configure Qiskit for using the credentials in
your IBM Q account:
여러분은 또한 Qiskit을 사용하여 여러분의 코드를 [real Quantum Chip](https://github.com/Qiskit/ibmqx-backend-information)에서 실행시킬 수 있습니다.

 먼저, API 토큰을 얻으세요:

-  [IBM Q experience](https://quantumexperience.ng.bluemix.net) 계정을 생성하세요.
-  IBM Q experience 웹 페이지에서 API token을 얻으세요. 토큰은 “My Account” > “Personal Access Token”에 위치해 있습니다.

이 API 토큰은 여러분의 프로그램을 IBM Q experience 백엔드 환경에서 실행할 수 있도록 해줄 것입니다. [예제](doc/example_real_backend.rst).

이에 대해 더 많은 정보를 알고 싶다면, 다음의 링크를 참조하세요. [our Qiskit documentation](doc/qiskit.rst).


### Next Steps

이제 여러분은 다른 [튜토리얼](https://github.com/Qiskit/qiskit-tutorial)들까지도 실행할 수 있는 준비와 환경 구성을 마쳤습니다. [index tutorial](https://github.com/Qiskit/qiskit-tutorial/blob/master/index.ipynb)을 시작하세요. 그리고 [‘Getting Started’ 예제](https://github.com/Qiskit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)로 이동하세요. 만일 여러분이 이미 [Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html)을 설치했다면, 여러분은 해당 노트북을 복사하고 수정하여 여러분 만의 노트북을 만들 수 있습니다.

튜토리얼을 Qiskit의 일부로서 설치하기 위해 다음의 링크를 확인하세요. [installation details](doc/install.rst#Install-Jupyter-based-tutorials). 완전한 SDK문서는 다음의 링크에서 확인할 수 있습니다. [*doc* directory](doc/qiskit.rst).

## More Information

다음은 Qiskit을 어떻게 써야하는지에 대한 더많은 정보와 튜토리얼 예제, 그리고 몇몇의 도움이 될만한 링크들 입니다. 한 번 살펴보세요.
* **[User Guides](https://github.com/Qiskit/ibmqx-user-guides)**,
  양자정보와 양자컴퓨팅에 대해 배울 수 있는 좋은 시작점
* **[Tutorials](https://github.com/Qiskit/qiskit-tutorial)**,
  예를 들어 Jupyter 노트북의 경우 [index](https://github.com/Qiskit/qiskit-tutorial/blob/master/index.ipynb)를 참고하세요. 그리고 [‘Getting Started’ Jupyter notebook](https://github.com/Qiskit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)도 함께 보세요.
* **[OpenQASM](https://github.com/Qiskit/openqasm)**,
  QASM에 대한 추가적인 정보와 예제
* **[IBM Quantum Experience Composer](https://quantumexperience.ng.bluemix.net/qx/editor)**,
  실제 양자컴퓨터 및 시뮬레이션된 양자컴퓨터와의 GUI 인터페이스를 통한 인터렉션
* **[QISkit Python API](https://github.com/Qiskit/qiskit-api-py)**, 파이썬을 통해 IBM Quantum Experience를 사용할 수 있는 API

Qiskit은 본래 [IBM Research](http://www.research.ibm.com/)연구팀과 [IBM-Q](http://www.research.ibm.com/ibm-q/)개발팀에 의해 양자컴퓨터에 대한 고수준(high level) 개발킷을 제공할 목적으로 개발되었습니다.
질의응답과 토론 그리고 양자컴퓨팅에 대해 더 넓게 살펴보기 위해 [IBM Q experience community](https://quantumexperience.ng.bluemix.net/qx/community)를 방문하세요. 만일 여러분이 Qiskit에 기여하길 원한다면 다음의 가이드라인을 살펴보세요. [contribution guidelines](.github/CONTRIBUTING.rst).

## Multilanguage guide

* **[Korean Translation](https://github.com/Qiskit/qiskit-terra/tree/master/doc/ko/README-ko.md)**, 한글 기본 가이드 라인

## Authors (alphabetical)

Qiskit was originally authored by
Luciano Bello, Jim Challenger, Andrew Cross, Ismael Faro, Jay Gambetta, Juan Gomez,
Ali Javadi-Abhari, Paco Martin, Diego Moreda, Jesus Perez, Erick Winston and Chris Wood.

And continues to grow with the help and work of [many people](https://github.com/Qiskit/qiskit-terra/graphs/contributors) who contribute
to the project at different levels.
