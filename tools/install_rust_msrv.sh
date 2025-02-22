#!/bin/sh
if [ ! -d rust-installer ]; then
    mkdir rust-installer
    wget https://sh.rustup.rs -O rust-installer/rustup.sh
    sh rust-installer/rustup.sh -y -c llvm-tools
    rustup default 1.79
    rustup component add llvm-tools
    rustup uninstall stable
fi
. "$HOME/.cargo/env"
