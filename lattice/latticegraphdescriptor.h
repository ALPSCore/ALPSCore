/***************************************************************************
* ALPS++/lattice library
*
* lattice/latticegraphdescriptor.h    the lattice graph descriptor class
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@comp-phys.orgh>
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

#ifndef ALPS_LATTICE_LATTICEGRAPHDESCRIPTOR_H
#define ALPS_LATTICE_LATTICEGRAPHDESCRIPTOR_H

#include <alps/config.h>
#include <alps/parameters.h>
#ifndef ALPS_WITHOUT_XML
# include <alps/parser/parser.h>
#endif
#include <alps/lattice/graph.h>
#include <alps/lattice/lattice.h>
#include <alps/lattice/latticegraph.h>
#include <alps/lattice/latticedescriptor.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/hypercubic.h>
#include <alps/lattice/coordinatelattice.h>
#include <alps/vectorio.h>

#include <iostream>

namespace alps {

class LatticeGraphDescriptor
  : public hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell>,std::vector<alps::StringValue> >, std::vector<alps::StringValue> >
{
public:
  typedef hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell>, std::vector<alps::StringValue> >, std::vector<alps::StringValue> > base_type;
  typedef lattice_traits<base_type>::unit_cell_type unit_cell_type;
  typedef lattice_traits<base_type>::offset_type offset_type;
  typedef lattice_traits<base_type>::cell_descriptor cell_descriptor;
  typedef lattice_traits<base_type>::vector_type vector_type;
  typedef lattice_traits<base_type>::basis_vector_iterator basis_vector_iterator;
  typedef lattice_traits<base_type>::cell_iterator cell_iterator; 
  typedef lattice_traits<base_type>::size_type size_type;
  typedef lattice_traits<base_type>::boundary_crossing_type boundary_crossing_type;

  LatticeGraphDescriptor() {}
  
#ifndef ALPS_WITHOUT_XML
  LatticeGraphDescriptor(const alps::XMLTag&, std::istream&, 
       const LatticeMap& = LatticeMap(), 
       const FiniteLatticeMap& = FiniteLatticeMap(), 
       const UnitCellMap& = UnitCellMap());

  void write_xml(std::ostream&, const std::string& = "") const;
#endif

  const std::string& name() const { return name_;}
  void set_parameters(const alps::Parameters&);
private:
  std::string name_, lattice_name_, unitcell_name_;
  bool lattice_is_finite_;
  FiniteLatticeDescriptor finitelattice_; 
  LatticeDescriptor lattice_; // for printing only
};

template<>
struct lattice_traits<LatticeGraphDescriptor>
{
  typedef LatticeGraphDescriptor::unit_cell_type unit_cell_type;
  typedef LatticeGraphDescriptor::cell_descriptor cell_descriptor;
  typedef LatticeGraphDescriptor::offset_type offset_type;
  typedef LatticeGraphDescriptor::basis_vector_iterator basis_vector_iterator;
  typedef LatticeGraphDescriptor::cell_iterator cell_iterator;
  typedef LatticeGraphDescriptor::size_type size_type;
  typedef LatticeGraphDescriptor::vector_type vector_type;
  typedef LatticeGraphDescriptor::boundary_crossing_type boundary_crossing_type;
};

typedef lattice_graph<LatticeGraphDescriptor,coordinate_graph_type> HypercubicLatticeGraphDescriptor;
typedef lattice_graph<hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > >, coordinate_graph_type> HypercubicLatticeGraph;

} // end namespace alps

#ifndef ALPS_WITHOUT_XML

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<< (std::ostream& out, const alps::LatticeGraphDescriptor& l)
  {
    l.write_xml(out);
    return out;
  }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif

#endif // ALPS_LATTICE_LATTICEGRAPHDESCRIPTOR_H
