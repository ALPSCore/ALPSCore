/***************************************************************************
* PALM++/lattice library
*
* example/example1.C
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
*                            Synge Todo <wistaria@comp-phys.org>
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#include <alps/lattice.h>
#include <iostream>

#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
using namespace alps;
#endif

int main() {
#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif
    typedef alps::coordinate_graph_type graph_type;

    // create the library from an XML file
    alps::LatticeLibrary lib(std::cin);

    // generate graph and set parity
    graph_type graph;
    alps::make_graph_from_lattice(graph, lib.lattice("square lattice 4x4"));
    alps::set_parity(graph);

    // write the library in XML
    std::cout << graph;

    for (graph_type::vertex_iterator vi = boost::vertices(graph).first;
	 vi != boost::vertices(graph).second; ++vi) {
      std::cout << "vertex " << *vi << "'s parity is ";
      if (boost::get(alps::parity_t(), graph, *vi)
	  == alps::parity::white) {
	std::cout << "white\n";
      } else if (boost::get(alps::parity_t(), graph, *vi)
		 == alps::parity::black) {
	std::cout << "black\n";
      } else {
	std::cout << "undefined\n";
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
}
