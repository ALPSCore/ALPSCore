#!/bin/bash
# Install prerequisites for TravisCI builds

set -ve

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
    sudo apt-get update
    sudo apt-get install -y --allow-unauthenticated g++-5
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
else
    brew cask uninstall oclint # TravisCI bug, see https://github.com/travis-ci/travis-ci/issues/8826#issuecomment-350103392
    brew install hdf5
    brew list
fi
