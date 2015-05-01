/** @file params_example.cpp
    @brief alps::params example 1
*/

#include <iostream>
#include "alps/params.hpp"

int main(int argc, const char* argv[])
{
  alps::params par(argc,argv);

  std::cout << par;
  return 0;
}
