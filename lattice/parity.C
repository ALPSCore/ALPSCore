/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2006 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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
#include <alps/parameter.h>
#include <fstream>
#include <iostream>

#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
using namespace alps;
#endif

int main() {
#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif
    typedef alps::coordinate_graph_type graph_t;
    typedef alps::parity_t parity_t;

    // read parameters
    alps::ParameterList parms;
    std::cin >> parms;
    
    for (alps::ParameterList::const_iterator p = parms.begin();
         p != parms.end(); ++p) {
      // create the lattice

      alps::graph_helper<> lattice(*p);
      const graph_t& graph = lattice.graph();
      
      std::cout << graph;
      
      for (graph_t::vertex_iterator vi = boost::vertices(graph).first;
           vi != boost::vertices(graph).second; ++vi) {
        std::cout << "vertex " << *vi << "'s parity is ";
        if (boost::get(parity_t(), graph, *vi)
            == alps::parity_traits<parity_t, graph_t>::white) {
          std::cout << "white\n";
        } else if (boost::get(alps::parity_t(), graph, *vi)
                   == alps::parity_traits<parity_t, graph_t>::black) {
          std::cout << "black\n";
        } else {
          std::cout << "undefined\n";
        }        
      }
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
