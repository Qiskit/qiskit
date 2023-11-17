#!/bin/sh
if [ ! -d rust-installer ]; then
    mkdir rust-installer
    wget https://sh.rustup.rs -O rust-installer/rustup.sh
    sh rust-installer/rustup.sh -y --default-toolchain 1.73
fi
