#!/bin/bash

# ------------------------------------------------------------------------------
# build_dependencies.sh
# 
# Dependency installer for qiskit-sdk-py/src/qasm-simulator-cpp
# Running the script:
#   1. Run the script directly
#     ./build_dependencies.sh  
#   2. Alternatively, build the make depend target (See the Makefile)
#     make depend
#
# .. note:: Tested on Ubuntu 16.04 only. 
# ------------------------------------------------------------------------------ 
set -ex
os_type=`uname -s`

# Check who is the current user.
USER=$(whoami)

if [ ${USER} == "root" ]
then
  SUDOCMD=""
else
  SUDOCMD="sudo"
fi

# Check the OS Type
echo "OS is $os_type"

if [[ "$os_type" == "Darwin" ]]; then
    ${SUDOCMD} xcode-select --install
elif [[ "$os_type" == "Linux" ]]; then
    echo "Installing dependencies on Linux"    
    linux_distro=`cat /etc/*release | grep "ID_LIKE=" | cut -c9- | tr -d '"'`
    if [[ "$linux_distro" == "debian" ]]; then
        ${SUDOCMD} apt-get update
        ${SUDOCMD} apt-get -y install build-essential libblas-dev liblapack-dev
    elif [[ "$linux_distro" == "fedora" ]]; then
        ${SUDOCMD} yum update
        ${SUDOCMD} yum -y install devtoolset-6 blas blas-devel lapack lapack-devel
        ${SUDOCMD} scl enable devtoolset-6 bash
    else
        echo "Unsupported linux distro: $linux_distro"
    fi
fi
