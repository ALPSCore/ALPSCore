Contributions are welcome!

The basic guidelines for contributing:

* GitHub is used as the collaborative environment.  You are welcome to
  clone [our repo](https://github.com/ALPSCore/ALPSCore) and submit a
  pull request.
* Bug reports and feature requests should be submitted to the
  project's [issue tracker](https://github.com/ALPSCore/ALPSCore/issues).
* The project uses CMake as its build system.
* Contributed code should be structured as described in
  [Repository structure](https://github.com/ALPSCore/ALPSCore/wiki/Repository-structure).
* Any new code should be accompanied by a unit test using
  [Google Test](https://github.com/google/googletest).
* The only allowed
  [dependencies](https://github.com/ALPSCore/ALPSCore/wiki/Installation#prerequisites)
  are [Boost](http://boost.org), HDF5 and MPI.
* Introducing dependencies on Boost libraries that are
  [not header-only](http://www.boost.org/doc/libs/1_61_0/more/getting_started/unix-variants.html#header-only-libraries)
  is discouraged unless really necessary.
* The language is C++03 to minimize incompatibilities with existing
  HPC (High Performance Computing) systems (if you really, really need
  to use a C++11 feature,
  [it can be discussed](https://github.com/ALPSCore/ALPSCore/issues)).
