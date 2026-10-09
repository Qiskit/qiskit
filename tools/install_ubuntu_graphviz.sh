#!/bin/sh

# Install a functional version of `graphviz` including the plugins we use.  Requires root.

set -e

apt upgrade
apt install -y graphviz libgvplugin-neato-layout8
