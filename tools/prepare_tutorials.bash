#!/bin/bash
#
# Clone the tutorials in from Qiskit/qiskit-tutorials, and put them in the right
# place in the documentation structure ready for a complete documentation build.
# In the initial transition from the metapackage structure, we are leaving
# placeholder files in the documentation directories so everything will build
# happily without the full structure having been cloned, but this may change in
# the future.
# 
# If the placeholder files are removed in the future, this script will likely
# have become obsolete and the CI pipelines (or this script) should be updated
# to reflect that.
#
# Usage:
#   prepare_tutorials.sh components ...
#
# components
#   Subdirectories of `tutorials` in the tutorials repository that should be
#   moved into the correct locations in the source tree.

set -e

if [[ $# -eq 0 ]]; then
    echo "Usage: prepare_tutorials.sh components ..." >&2
    exit 1
fi

# Pull in the tutorials repository.
tmpdir="$(mktemp -d)"
git clone --depth=1 https://github.com/Qiskit/qiskit-tutorials "$tmpdir"
indir="${tmpdir}/tutorials"

outdir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/docs/tutorials"
if [[ ! -d "$outdir" ]]; then
    echo "Tutorials documentation directory '${outdir}' does not exist." >&2
    exit 2
fi

for component in "$@"; do
    echo "Getting tutorials from '${component}'"

    if [[ ! -d "${indir}/${component}" ]]; then
        echo "Component '${component}' not in tutorials repository." >&2
        exit 3
    fi
    if [[ -d "${outdir}/${component}" && -f "${outdir}/${component}/placeholder.ipynb" ]]; then
        rm "${outdir}/${component}/placeholder.ipynb"
        if [[ -z "$(ls -A "${outdir}/${component}")" ]]; then
            rm -r "${outdir}/${component}"
        else
            echo "Directory '${outdir}/${component}' contains files other than the placeholder. This script needs updating." >&2
            exit 4
        fi
    else
        echo "Directory '${outdir}/${component}' does not exist, or has no placeholder. This script needs updating." >&2
        exit 5
    fi
    mv "${indir}/${component}" "${outdir}/${component}"
done

rm -rf "${tmpdir}"
