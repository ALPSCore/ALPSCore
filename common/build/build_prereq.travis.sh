#!/bin/bash
# Install prerequisites for TravisCI builds

set -ve

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
    # Jammy's default toolchain (GCC 11) is sufficient; packages come from .travis.yml
    true
else
    brew install hdf5
    brew list
fi
