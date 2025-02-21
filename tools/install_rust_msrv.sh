#!/bin/sh
if [ ! -d rust-installer ]; then
    mkdir rust-installer
    wget https://sh.rustup.rs -O rust-installer/rustup.sh
    sh rust-installer/rustup.sh -y -c llvm-tools -t 1.79-aarch64-unknown-linux-gnu
fi
. "$HOME/.cargo/env"
