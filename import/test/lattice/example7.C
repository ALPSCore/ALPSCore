/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2004-2008 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Ian McCulloch <ianmcc@physics.uq.edu.au>
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
#include <alps/lattice/unitcell.h>
#include <boost/foreach.hpp>
#include <iostream>
#include <fstream>

#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
using namespace alps;
#endif

template <typename FwdIter>
std::string PrintVector(FwdIter start, FwdIter finish)
{
   std::ostringstream out;
   out << '(';
   if (start != finish)
   {
      out << *start;
      ++start;
   }
   while (start != finish)
   {
      out << ',' << *start;
      ++start;
   }
   out << ')';
   return out.str();
};

template <typename Container>
inline
std::string PrintVector(Container const& c)
{
   return PrintVector(c.begin(), c.end());
}

void ShowUnitCell(alps::GraphUnitCell const& unitcell)
{
  typedef alps::GraphUnitCell::graph_type graph_type;
  typedef alps::type_type type_type;                   // this is int
  typedef alps::coordinate_type coordinate_type;       // this is std::vector<double>

  typedef alps::graph_traits<graph_type>::site_iterator   site_iterator;
  typedef alps::graph_traits<graph_type>::site_descriptor site_descriptor;
  typedef alps::graph_traits<graph_type>::bond_iterator   bond_iterator;
  typedef alps::graph_traits<graph_type>::bond_descriptor bond_descriptor;

  typedef std::vector<int> offset_type;

  graph_type const& graph = unitcell.graph();

  // property map for the site type (see example5)
  alps::property_map<alps::site_type_t, graph_type, type_type>::const_type
    site_type(alps::get_or_default(alps::site_type_t(), graph, type_type()));

  // property map for the coordinates of each site (see example5)
  alps::property_map<alps::coordinate_t, graph_type, coordinate_type>::const_type
    site_coordinate(alps::get_or_default(alps::coordinate_t(), graph, coordinate_type()));

  // property map for the bond type (see example6)
  alps::property_map<alps::bond_type_t, graph_type, type_type>::const_type
    bond_type(alps::get_or_default(alps::bond_type_t(), graph, type_type()));

  // property map for the cell offset for the source of a bond
  alps::property_map<alps::source_offset_t, graph_type, offset_type>::const_type
    source_offset(alps::get_or_default(alps::source_offset_t(), graph, offset_type()));

  // property map for the cell offset for the target of a bond
  alps::property_map<alps::target_offset_t, graph_type, offset_type>::const_type
    target_offset(alps::get_or_default(alps::target_offset_t(), graph, offset_type()));

  // determine the total number of sites in the unit cell
  std::cout << "The unit cell has " << num_sites(graph) << " sites.\n";

  std::cout << "The sites in the unit cell are:\n";
  site_iterator site_it, site_end;
  for (boost::tie(site_it, site_end) = sites(graph); site_it != site_end;
       ++site_it)
  {
    site_descriptor site = *site_it;
    type_type type = site_type[site];
    coordinate_type coord = site_coordinate[site];

    std::cout << "Site " << site
              << " has type " << type << " and coordinates " << PrintVector(coord) << '\n';
  }

  std::cout << "The bonds in the unit cell are:\n";
  bond_iterator bond_it, bond_end;
  for (boost::tie(bond_it, bond_end) = bonds(graph); bond_it != bond_end;
       ++bond_it)
  {
    bond_descriptor bond = *bond_it;
    site_descriptor source = boost::source(bond, graph);
    site_descriptor target = boost::target(bond, graph);

    offset_type soffset = source_offset[bond];
    offset_type toffset = target_offset[bond];
    type_type type = bond_type[bond];

    std::cout << "The bond between site " << source << " offset " << PrintVector(soffset)
              << " and site " << target << " offset " << PrintVector(toffset)
              << " has type " << type << std::endl;
  }
  std::cout << std::endl;
}

template <typename LatticeType>
void IterateOverCells(const LatticeType& lattice)
{
  // iterate over all cells in the lattice.

  // In the previous examples for the sites and bonds, the iterator types
  // are properties of the graph, accessed through alps::graph_traits.
  // But the cell is a property of the lattice and is accessed through alps::lattice_traits.
  typedef typename alps::lattice_traits<LatticeType>::cell_iterator cell_iterator;

  // Each cell as an associated integer 'unit_cell_type', acessed through
  // the unit_cell_type_t property tag.
  // This gives access to the 'type' attribute of the <EDGE> xml element.
  //  typename alps::property_map<alps::bond_type_t, LatticeType ,int>::const_type
  //    bond_type(get_or_default(alps::bond_type_t(), lattice, 0));

  // determine the total number of cells in the lattice
  //  std::cout << "The lattice comprises " << num_bonds(lattice) << " bonds.\n";

  typedef typename alps::lattice_traits<LatticeType>::offset_type offset_type; // a vector of integers
  typedef typename alps::lattice_traits<LatticeType>::vector_type vector_type; // a vector of reals
  typedef typename alps::lattice_traits<LatticeType>::size_type size_type; // size_type-me-harder

  std::cout << "The basis vectors are: ";
  BOOST_FOREACH(vector_type const& v, basis_vectors(lattice))
    std::cout << PrintVector(v) << ' ';
  std::cout << std::endl;

  std::cout << "The extent of the lattice is: "
            << PrintVector(lattice.extent()) << '\n';

  std::cout << "The lattice spanning vectors are: ";
  int i = 0;
  BOOST_FOREACH(vector_type const& v, basis_vectors(lattice)) {
    vector_type sv(v);
    BOOST_FOREACH(double& x, sv) x *= lattice.extent(i);
    std::cout << PrintVector(sv) << ' ';
    ++i;
  }
  std::cout << std::endl;

  std::cout << "The number of cells in the lattice is "
            << volume(lattice) << '\n';
  std::cout << "The cells in the lattice are:\n";

  // cells(lattice) returns a [begin, end) pair of iterators over all cells
  cell_iterator cell_it, cell_end;
  for (boost::tie(cell_it, cell_end) = cells(lattice); cell_it != cell_end;
       ++cell_it)
  {
     // the cell index
     size_type Index = index(*cell_it, lattice);
     std::cout << "index=" << Index;

     // the cell offset, as a function of the basis vectors
     offset_type Offset = offset(*cell_it, lattice);
     std::cout << ", offset=" << PrintVector(Offset);

     // We can also obtain the cell_descriptor corresponding to an offset
     assert(Offset == offset(cell(Offset, lattice), lattice));

     // the coordinates of the center of the unit cell
     vector_type CellOrigin(unit_cell(lattice).dimension(), 0);
       // the zero-vector
     vector_type Coordinates = alps::coordinate(*cell_it, CellOrigin, lattice);
     std::cout << ", coordinates_of_center_of_cell=" << PrintVector(Coordinates);

     std::cout << '\n';
  }
  std::cout << std::endl;
}

int main()
{

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

  // read parameters
  alps::ParameterList plist(std::cin);
  BOOST_FOREACH(alps::Parameters const& p, plist) {
    // create a graph factory with default graph type
    alps::graph_helper<> lattice(p);

    ShowUnitCell(lattice.unit_cell());

    IterateOverCells(lattice.lattice());
  }

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
