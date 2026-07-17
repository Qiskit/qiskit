#!/bin/sh
set -e

if [ ! -d rust-installer ]; then
    mkdir rust-installer
    wget https://sh.rustup.rs -O rust-installer/rustup.sh
    msrv="$(python3 <<EOF
from pathlib import Path
import tomllib

cargo_toml_file = Path("$0").absolute().parents[1] / "Cargo.toml"
manifest = tomllib.load(open(cargo_toml_file, 'rb'))
print(manifest["workspace"]["package"]["rust-version"])
EOF
)"
    sh rust-installer/rustup.sh -y --default-toolchain "$msrv" --component llvm-tools
fi
. "$HOME/.cargo/env"
