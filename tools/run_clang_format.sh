#! /bin/bash
set -e

if [ -z "$1" ]; then
    args="--dry-run --Werror"  # do a dry run
elif [ "$1" = "apply" ]; then
    args="-i"  # inplace change of the files
else
    # any other argument is invalid
    echo "Invalid argument, either no arguments or 'apply' is supported, not: $1"
    exit 1
fi

# this is the style file 
our_dir="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
repo_root="$(realpath -- "$our_dir/..")"
style="${repo_root}/.clang-format"

# get all tracked files in HEAD, and filter for files ending in .c or .h
files=$(git ls-files $repo_root | grep ".*\.[c,h]$")
echo $files

# apply clang format on all files
status=0
for file in $files; do
    # we don't want to exit prematurely but process all files
    if ! clang-format --style="file:$style" $args $file; then
        status=1
    fi
done
exit $status
