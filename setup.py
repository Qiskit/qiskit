from setuptools import setup

packages = ["qiskit",
            "qiskit.backends",
            "qiskit.dagcircuit",
            "qiskit.extensions",
            "qiskit.extensions.standard",
            "qiskit.mapper",
            "qiskit.qasm",
            "qiskit.qasm._node",
            "qiskit.unroll",
            "qiskit.tools",
            "qiskit.tools.apps",
            "qiskit.tools.qcvv",
            "qiskit.tools.qi"]

requirements = ["IBMQuantumExperience>=1.8.13",
                "requests>=2.18,<2.19",
                "networkx>=1.11,<1.12",
                "ply==3.10",
                "numpy>=1.13,<1.14",
                "scipy>=0.19,<0.20",
                "matplotlib>=2.0,<2.1",
                "sphinx>=1.6,<1.7"]

setup(
    name="qiskit",
    version="0.4.0",
    description="Software for developing quantum computing programs",
    long_description="""QISKit is a software development kit for writing
        quantum computing experiments, programs, and applications. Works with
        Python 3.5 and 3.6""",
    url="https://github.com/QISKit/qiskit-sdk-py",
    author="QISKit Development Team",
    author_email="qiskit@us.ibm.com",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=packages,
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5",
)
