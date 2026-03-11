#!/bin/bash

set -e

repo_root="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
cd "$repo_root"

target_dir="$( \
    cargo metadata --no-deps --format-version=1 \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])' \
)"
cargo build --quiet -p qiskit-bindgen-c
"$target_dir/debug/qiskit-bindgen-c" lint-slots --cext-path "$repo_root/crates/cext"
