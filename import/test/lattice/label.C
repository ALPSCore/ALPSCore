/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2006 by Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id$ */

#include <alps/lattice.h>
#include <iostream>

#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
using namespace alps;
#endif

int main()
{

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

    // read parameters
    alps::Parameters parameters;
    std::cin >> parameters;
    // create a graph factory with default graph type
    alps::graph_helper<> lattice(parameters);

    std::vector<std::string> label;

    // site labels
    std::cout << "Site labels:\n";
    label = lattice.site_labels();
    for (std::vector<std::string>::const_iterator itr = label.begin();
         itr != label.end(); ++itr)
      std::cout << *itr << std::endl;

    // bond labels
    std::cout << "Bond labels:\n";
    label = lattice.bond_labels();
    for (std::vector<std::string>::const_iterator itr = label.begin();
         itr != label.end(); ++itr)
      std::cout << *itr << std::endl;

    // momenta label
    std::cout << "Momenta labels:\n";
    label = lattice.momenta_labels(6);
    for (std::vector<std::string>::const_iterator itr = label.begin();
         itr != label.end(); ++itr)
      std::cout << *itr << std::endl;

    // distance label
    std::cout << "Distance labels:\n";
    label = lattice.distance_labels();
    for (std::vector<std::string>::const_iterator itr = label.begin();
         itr != label.end(); ++itr)
      std::cout << *itr << std::endl;
  
#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& e)
{
  std::cerr << "Caught exception: " << e.what() << "\n";
  exit(-1);
}
catch (...)
{
  std::cerr << "Caught unknown exception\n";
  exit(-2);
}
#endif
  return 0;
}
