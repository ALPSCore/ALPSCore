Contributions are welcome!

The basic guidelines for contributing:

* GitHub is used as the collaborative environment. You are welcome to clone [our repo](https://github.com/ALPSCore/ALPSCore) and submit a pull request.
* Bug reports and feature requests should be submitted to the project's [issue tracker](https://github.com/ALPSCore/ALPSCore/issues).
* The project uses [CMake](https://cmake.org/) (3.16 or newer) as its build system.
* Contributed code should be structured as described in [Repository structure](https://github.com/ALPSCore/ALPSCore/wiki/Repository-structure).
* Any new code should be accompanied by a unit test using [Google Test](https://github.com/google/googletest).
* The only allowed [dependencies](https://github.com/ALPSCore/ALPSCore/wiki/Installation#prerequisites) are [**header-only**](http://www.boost.org/doc/libs/release/more/getting_started/unix-variants.html#header-only-libraries) [Boost](http://boost.org), HDF5 and MPI.
* The language of the libraries is C++11 to minimize incompatibilities with existing HPC (High Performance Computing) systems (if you really, really need to use a C++14 or newer feature,  [it can be discussed](https://github.com/ALPSCore/ALPSCore/issues)). The unit tests are compiled as C++17, because the bundled Google Test requires it, so building them (`Testing=ON`, the default) needs a C++17 compiler; with an older compiler, configure with `-DTesting=OFF`.
