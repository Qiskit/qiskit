# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import os
import platform
from distutils.command.build import build
from multiprocessing import cpu_count
from subprocess import call

from setuptools import setup, find_packages
from setuptools.dist import Distribution


requirements = [
    "jsonschema>=2.6,<2.7",
    "marshmallow>=2.16.3,<3",
    "marshmallow_polyfield>=3.2,<4",
    "networkx>=2.2",
    "numpy>=1.13",
    "pillow>=4.2.1",
    "ply>=3.10",
    "psutil>=5",
    "requests>=2.19",
    "requests-ntlm>=1.1.0",
    "scipy>=0.19,!=0.19.1",
    "sympy>=1.3"
]


# C++ components compilation
class QasmSimulatorCppBuild(build):
    def run(self):
        super().run()
        # Store the current working directory, as invoking cmake involves
        # an out of source build and might interfere with the rest of the steps.
        current_directory = os.getcwd()

        try:
            supported_platforms = ['Linux', 'Darwin', 'Windows']
            current_platform = platform.system()
            if current_platform not in supported_platforms:
                # TODO: stdout is silenced by pip if the full setup.py invocation is
                # successful, unless using '-v' - hence the warnings are not printed.
                print('WARNING: Qiskit cpp simulator is meant to be built with these '
                      'platforms: {}. We will support other platforms soon!'
                      .format(supported_platforms))
                return

            cmd_cmake = ['cmake', '-vvv']
            if 'USER_LIB_PATH' in os.environ:
                cmd_cmake.append('-DUSER_LIB_PATH={}'.format(os.environ['USER_LIB_PATH']))
            if current_platform == 'Windows':
                # We only support MinGW so far
                cmd_cmake.append("-GMinGW Makefiles")
            cmd_cmake.append('..')

            cmd_make = ['make', 'pypi_package_copy_qasm_simulator_cpp']

            try:
                cmd_make.append('-j%d' % cpu_count())
            except NotImplementedError:
                print('WARNING: Unable to determine number of CPUs. Using single threaded make.')

            def compile_simulator():
                self.mkpath('out')
                os.chdir('out')
                call(cmd_cmake)
                call(cmd_make)

            self.execute(compile_simulator, [], 'Compiling C++ QASM Simulator')
        except Exception as e:
            print(str(e))
            print("WARNING: Seems like the cpp simulator can't be built, Qiskit will "
                  "install anyway, but won't have this simulator support.")

        # Restore working directory.
        os.chdir(current_directory)


# This is for creating wheel specific platforms
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="qiskit-terra",
    version="0.7.0",
    description="Software for developing quantum computing programs",
    long_description="""Terra provides the foundations for Qiskit. It allows the user to write 
        quantum circuits easily, and takes care of the constraints of real hardware.""",
    url="https://github.com/Qiskit/qiskit-terra",
    author="Qiskit Development Team",
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
    packages=find_packages(exclude=['test*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5",
    cmdclass={
        'build': QasmSimulatorCppBuild,
    },
    distclass=BinaryDistribution,
    extra_requires={
        'visualization': ['matplotlib>=2.1', 'nxpd>=0.2', 'ipywidgets>=7.3.0',
                          'pydot'],
        'full-featured-simulators': ['qiskit-aer>=0.1']
    }
)
