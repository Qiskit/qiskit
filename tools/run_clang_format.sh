#! /bin/bash

if [ -z "$1" ]; then
    args="--dry-run --Werror"  # do a dry run
elif [ "$1" = "apply" ]; then
    args="-i"  # inplace change of the files
else
    # any other argument is invalid
    echo "Invalid argument, either no arguments or 'apply' is supported, not: $1"
    exit 1
fi

# this is the style file -- note that this script should
# be run from root, such that this file is correctly found
style=".clang-format"

# get all tracked files in HEAD, and filter for files ending in .c or .h
files=$(git ls-tree --name-only -r HEAD | grep ".*\.[c,h]$")

# apply clang format on all files
for file in $files; do
    clang-format --style="file:$style" $args $file
done
