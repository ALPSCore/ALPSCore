/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2004-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Ian McCulloch <ianmcc@physik.rwth-aachen.de>
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
#include <fstream>

#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
using namespace alps;
#endif

template <typename GraphType>
void IterateOverSites(const GraphType& graph)
{
  // iterate over all sites in the graph, using site_iterator's.

  // The iterator type is obtained from alps::graph_traits
  typedef typename alps::graph_traits<GraphType>::site_iterator site_iterator;

  // The sites in the lattice are denoted using a type site_descriptor.
  // For a generic boost graph this type depends on the internal storage
  // of the graph, however in ALPS it is always an integer and
  // corresponds to the "id" atribute of the <VERTEX> xml element.
  typedef typename alps::graph_traits<GraphType>::site_descriptor site_descriptor;
  BOOST_STATIC_ASSERT((boost::is_convertible<site_descriptor, int>::value));

  // We can access some properties of the sites by using a
  // property_map from the boost graph library.
  // Different properties are accessed using a tag, the site_type_t property tag 
  // corresponds to the "type" attribute of the <VERTEX> xml element.
  // If GraphType doesn't have a site_type_t property, then
  // it defaults to an int (third template parameter to 
  // alps::property_map<>) with value 0.
  typename alps::property_map<alps::site_type_t, GraphType, int>::const_type
    site_type(alps::get_or_default(alps::site_type_t(), graph, 0));

  // the coordinate_t property tag gives access to the coordinates of the
  // site in the lattice.  The coordinates are represented as std::vector<double>.
  // FIXME: should that be std::vector<int> ?
  typename alps::property_map<alps::coordinate_t, GraphType, std::vector<double> >::const_type
    site_coordinate(alps::get_or_default(alps::coordinate_t(), graph, std::vector<double>()));

  // determine the total number of sites in the graph
  std::cout << "The graph has " << num_sites(graph) << " sites.\n";

  // sites(graph) returns a [begin, end) pair of iterators over all sites
  site_iterator site_it, site_end;
  for (boost::tie(site_it, site_end) = sites(graph); site_it != site_end;
       ++site_it) 
  {
    site_descriptor site = *site_it;

    int type = site_type[site];
    std::vector<double> coordinates = site_coordinate[site];

    std::cout << "Site " << site
              << " has type " << type;
    if (coordinates.empty())
      std::cout << " and has no coordinates defined";
    else
    {
      std::cout << " and has coordinates ";
      std::copy(coordinates.begin(), 
                coordinates.end(), 
                std::ostream_iterator<double>(std::cout, " "));
    }
    std::cout << std::endl;
  }
}

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

    IterateOverSites(lattice.graph());

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
