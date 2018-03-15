# Quantum Information Software Kit (QISKit)

[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/QISKit/qiskit-sdk-py.svg?branch=master)](https://travis-ci.org/QISKit/qiskit-sdk-py)

O Quantum Information Software Kit (**QISKit**) é um kit de desenvolvimento de software (SDK) para trabalhar com o [OpenQASM](https://github.com/QISKit/qiskit-openqasm) e com a [IBM Q experience (QX)](https://quantumexperience.ng.bluemix.net/).

Utilize o **QISKit** para criar, compilar e executar programas de computação quântica em um dos diversos back-ends (processadores quânticos reais online, simuladores online e simuladores locais). Para os back-ends online, QISKit usa um [cliente de uma API em python](https://github.com/QISKit/qiskit-api-py) para conectar-se à IBM Q experience.

**Usamos o GitHub issues para estar pendentes de requests e bugs. Para todo tipo de pergunta ou discussão, favor acessar a** [comunidade IBM Q experience](https://quantumexperience.ng.bluemix.net/qx/community).

**Se você deseja contribuir ao QISKit, por favor dê uma olhada nos nossos** [guias de contribuição](CONTRIBUTING.rst).

Links para as seções:

* [Instalação](#installation)
* [Criando o seu primeiro Programa Quântico](#creating-your-first-quantum-program)
* [Mais Informações](#more-information)
* [Autores](#authors-alphabetical)
* [Licença](#license)

## Instalação

### Dependências

É necessário possuir a [versão 3.5 do Python](https://www.python.org/downloads/) (ou outra mais recente) para usar o QISKit. Também se recomenda utilizar o [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) para interagir com os tutoriais.
Por essa razão, recomendamos a instalação da distribuição [Anaconda 3](https://www.continuum.io/downloads) do python, já que ela traz todas as dependências pré-instaladas.

Além disso, um entendimento básico de informação quântica é muito útil ao interagir com o QISKit. Se você é novo no tema, comece com os nossos [Guias de Usuário](https://github.com/QISKit/ibmqx-user-guides)!


### Instalação

Recomendamos instalar o QISKit com a ferramenta PIP (um gerenciador de pacotes de python): 

```
    pip install qiskit
```
O PIP se encarregará de todas as dependências automaticamente e você sempre terá a versão mais recente e bem testada.

O pacote PIP vem com os binários para as seguintes plataformas:

* Linux x86_64
* Darwin
* Win64

Se a sua plataforma não está na lista, PIP tentará realizar o build no momento da instalação. É necessário ter o CMake 3.5 ou uma versão mais recente pré-instalada, e ao menos um dos [ambientes de build suportados pelo CMake](https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html).

Se durante a instalação o PIP falhar em realziar o build, não se preocupe, no final você terá o QISKit instalado, mas provavelmente você não poderá tirar vantagem de alguns dos componentes de alta performance. De qualquer forma, nós sempre proporcionamos uma opção não tão rápida em python como última alternativa.


### Configure o seu ambiente

Nós recomendamos a utilização de um ambiente virtual em pyhton para melhorar a sua experiencia. Consulte nossa [documentação para Configurar um Ambiente](doc/install.rst#3.1-Setup-the-environment) para mais informações.

## Crie o seu primeiro Programa Quântico

Agora que o SDK foi instalado, é hora de começar a trabalhar com o QISKit.

Estamos prontos para testar um exemplo de circuito quântico que é executado no simulador local.

Este é um exemplo que realiza um entrelaçamento quântico.

```python
from qiskit import QuantumProgram, QISKitError, RegisterSizeError

# Create a QuantumProgram object instance.
q_program = QuantumProgram()
backend = 'local_qasm_simulator'
try:
    # Create a Quantum Register called "qr" with 2 qubits.
    quantum_reg = q_program.create_quantum_register("qr", 2)
    # Create a Classical Register called "cr" with 2 bits.
    classical_reg = q_program.create_classical_register("cr", 2)
    # Create a Quantum Circuit called "qc" involving the Quantum Register "qr"
    # and the Classical Register "cr".
    quantum_circuit =
        q_program.create_circuit("bell", [quantum_reg],[classical_reg])

    # Add the H gate in the Qubit 0, putting this qubit in superposition.
    quantum_circuit.h(quantum_reg[0])
    # Add the CX gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state
    quantum_circuit.cx(quantum_reg[0], quantum_reg[1])

    # Add a Measure gate to see the state.
    quantum_circuit.measure(quantum_reg, classical_reg)

    # Compile and execute the Quantum Program in the local_qasm_simulator.
    result = q_program.execute(["bell"], backend=backend, shots=1024, seed=1)

    # Show the results.
    print(result)
    print(result.get_data("bell"))

except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
except RegisterSizeError as ex:
    print('Error in the number of registers!. Error = {}'.format(ex))
```

Neste caso, o output será:

```
COMPLETED
{'counts': {'00': 512, '11': 512}}
```

Este script está disponível [aqui](examples/python/hello_quantum.py).

### Executando seu código em um chip quântico real

Você também pode utilizar o QISKit para executar seu código em um [chip quântico real](https://github.com/QISKit/ibmqx-backend-information).
Para tanto, você precisa configurar o SDK para usar as credenciais da sua Quantum Experience Account:


#### Configure o seu API token e as credenciais QE

1. Crie um conta em [IBM Q experience](https://quantumexperience.ng.bluemix.net)> se você ainda não possuir uma.
2. Consiga um API token no site da IBM Q experience em "`My Account`" > "`Personal Access Token`". Este token permite a execução de programas nos back-ends da IBM Q experience. [Exemplo](doc/example_real_backend.rst).
3. Vamos criar um novo arquivo chamado `Qconfig.py` e inserir nele o API token. O arquivo deve ter o seguinte conteúdo:
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
4. Substitua `'MY_API_TOKEN'` com o seu API token adquirido no passo 2.

5. Se você tem acesso aos IBM Q features, você precisa configurar também os valores para o seu hub, grupo e projeto. Para isso, assigne à variável `config` os valores que você pode encontrar na página da sua conta IBM Q.

Uma vez que o arquivo `Qconfig.py` já foi configurado, você deve colocá-lo no mesmo diretório/pasta do seu programa/tutorial, para que as configurações sejam importadas e utilizadas para autenticar com a função `Qconfig.py`. Por exemplo:

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

Para mais detalhes sobre esse exemplo e mais informações veja a [documentação para o QISKit](https://www.qiskit.org/documentation/).


### Próximos passos

Agora você está preparado para conferir alguns dos outros exemplos do nosso repositório [Tutorial](https://github.com/QISKit/qiskit-tutorial). Comece com o [index tutorial](https://github.com/QISKit/qiskit-tutorial/blob/master/index.ipynb) e depois siga com o [‘Getting Started’ example](https://github.com/QISKit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb).
Se você já tem [Notebooks Jupyter instalados](https://jupyter.readthedocs.io/en/latest/install.html), você pode copiar e modificar os notebooks para criar seus próprios experimentos.

Para instalar os tutoriais como parte do QISKit SDK, veja os seguintes [detalhes de instalação](doc/install.rst#Install-Jupyter-based-tutorials). A documentação completa do SDK pode ser encontrada na [pasta *doc*](doc/qiskit.rst) e no [site oficial do QISKit](https://www.qiskit.org/documentation).

## Mais Informações

Para mais informações sobre como usar o QISKit, tutoriais de exemplos e outros links úteis, dê uma olhada nos seguintes materiais:

* **[User Guides](https://github.com/QISKit/ibmqx-user-guides)**, um bom lugar para começar a aprender sobre informação quântica e computação quântica
* **[Tutorials](https://github.com/QISKit/qiskit-tutorial)**, para exemplos de notebooks, comece com o [index](https://github.com/QISKit/qiskit-tutorial/blob/master/index.ipynb) e [‘Getting Started’ Jupyter notebook](https://github.com/QISKit/qiskit-tutorial/blob/002d054c72fc59fc5009bb9fa0ee393e15a69d07/1_introduction/getting_started.ipynb)
* **[OpenQASM](https://github.com/QISKit/openqasm)**, para informações adicionais e exemplos de código QASM
* **[IBM Quantum Experience Composer](https://quantumexperience.ng.bluemix.net/qx/editor)**, uma interface gráfica com computadores quânticos reais e simulados.
* **[QISkit Python API](https://github.com/QISKit/qiskit-api-py)**, uma API para usar o IBM Quantum Experience em Python

O QISKit for desenvolvido originalmente por pesquisadores e desenvolvedores do [IBM-Q](http://www.research.ibm.com/ibm-q/) Team no [IBM Research](http://www.research.ibm.com/), com o objetivo de oferecer um kit de desenvolvimento em alto nível para trabalhar com computadores quânticos.

Visite a [comunidade IBM Q experience](https://quantumexperience.ng.bluemix.net/qx/community) para perguntas e discussões sobre o QISKit e computação quântica em geral. Se você quiser contribuir ao QISKit, dê uma olhada nas nossas [diretrizes de contribuição](CONTRIBUTING.rst).

## Guia em outros idiomas

* **[Coreano](doc/ko/README.md)**, Guia básico escrito em coreano.
* **[Chinês](doc/zh/README.md)**, Guia básico escrito em chinês.
* **[Inglês](README.md)**, Guia básico escrito em inglês.

## Autores (em ordem alfabética)

Ismail Yunus Akhalwaya, Jim Challenger, Andrew Cross, Stefan Devai, Vincent Dwyer, Mark Everitt, Ismael Faro, Jay Gambetta, Juan Gomez, Yunho Maeng, Paco Martin, Antonio Mezzacapo, Diego Moreda, Jesus Perez, Russell Rundle, Todd Tilma, John Smolin, Erick Winston, Chris Wood.

Em lançamentos futuros, qualquer um que contribua com código para o projeto é bem-vindo a incluir seu nome na lista.

## Licença

Este projeto usa a [licença de software Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
