#!/bin/bash

DOC_SOURCE_PATH=$1

if [[ $DOC_SOURCE_PATH != "./." ]] ; then
    exit
fi

REPO_PATH=`mktemp -d`

git clone https://github.com/Qiskit/qiskit-tutorial.git $REPO_PATH

cp -r $REPO_PATH/qiskit/basics/* $DOC_SOURCE_PATH/qiskit-tutorials/qiskit/basics/.
cp -r $REPO_PATH/qiskit/basics/* $DOC_SOURCE_PATH/qiskit-tutorials/qiskit/terra/.

rm -rf $REPO_PATH
