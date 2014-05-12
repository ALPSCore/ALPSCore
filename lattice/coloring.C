/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2009 by Synge Todo <wistaria@comp-phys.org>
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

// site coloring test program

#include <alps/lattice.h>
#include <alps/parameter.h>
#include <boost/graph/sequential_vertex_coloring.hpp>
#include <iostream>

#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
using namespace alps;
#endif

int main() {
#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

    typedef alps::graph_helper<> lattice_type;
    typedef lattice_type::graph_type graph_type;
    typedef boost::property_map<graph_type, alps::site_index_t>::const_type vertex_index_map;
    alps::ParameterList params(std::cin);
    BOOST_FOREACH(alps::Parameters const& p, params) {
      lattice_type lattice(p);
      std::vector<std::size_t> color(lattice.num_sites());
      int nc = boost::sequential_vertex_coloring(lattice.graph(),
        boost::iterator_property_map<std::size_t*, vertex_index_map>(&color.front(),
          get(boost::vertex_index, lattice.graph())));
      std::cout << "LATTICE = " << p["LATTICE"] << std::endl;
      std::cout << "  number of colors = " << nc << std::endl;
      std::cout << "  site colors =";
      for (int s = 0; s < lattice.num_sites(); ++s) std::cout << ' ' << color[s];
      std::cout << std::endl;
    }

#ifndef BOOST_NO_EXCEPTIONS
  }
  catch (std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << "\n";
    exit(-1);
  }
  catch (...) {
    std::cerr << "Caught unknown exception\n";
    exit(-2);
  }
#endif
  return 0;
}
