import sys
import os
from setuptools import setup
from setuptools.command.install import install
from distutils.command.build import build
from multiprocessing import cpu_count
from subprocess import call
import platform


requirements = [
    "IBMQuantumExperience>=1.8.26",
    "matplotlib>=2.1,<2.2",
    "networkx>=1.11,<2.1",
    "numpy>=1.13,<1.15",
    "ply==3.10",
    "scipy>=0.19,<1.1",
    "Sphinx>=1.6,<1.7",
    "sympy>=1.0"
]


packages = ["qiskit",
            "qiskit.backends",
            "qiskit.dagcircuit",
            "qiskit.extensions",
            "qiskit.extensions.standard",
            "qiskit.extensions.qiskit_simulator",
            "qiskit.extensions.quantum_initializer",
            "qiskit.mapper",
            "qiskit.qasm",
            "qiskit.qasm._node",
            "qiskit.unroll",
            "qiskit.tools",
            "qiskit.tools.apps",
            "qiskit.tools.qcvv",
            "qiskit.tools.qi"]


# C++ components compilation
class QiskitSimulatorBuild(build):
    def run(self):
        build.run(self)
        supported_platforms = ['Linux', 'Darwin']
        if not platform.system() in supported_platforms:
            print('WARNING: QISKit cpp simulator is ment to be built with these '
                  'platforms: {}. We will support other platforms soon!'
                  .format(supported_platforms))
            return

        target_platform = '{}-{}'.format(platform.system(), platform.machine()).lower()
        build_path = os.path.join(os.path.abspath(self.build_base), target_platform)
        binary_path = os.path.join(build_path, 'qiskit_simulator')

        cmd = [
            'make',
            'sim',
            'OUTPUT_DIR=' + build_path,
        ]

        try:
            cmd.append('-j%d' % cpu_count())
        except NotImplementedError:
            print('WARNING: Unable to determine number of CPUs. Using single threaded make.')

        def compile():
            call(cmd)

        try:
            self.execute(compile, [], 'Compiling QISKit C++ Simulator')
        except:
            print("WARNING: Seems like the cpp simulator can't be built, Qiskit will "
                  "install anyway, but won't have this simulator support.")
            return

        self.mkpath(self.build_lib)
        if not self.dry_run:
            self.copy_file(binary_path, '{}/qiskit/backends'.format(self.build_lib))

setup(
    name="qiskit",
    version="0.4.6",
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
    cmdclass={
        'build': QiskitSimulatorBuild,
    }
)
