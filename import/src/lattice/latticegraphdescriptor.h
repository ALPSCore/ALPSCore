/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@comp-phys.orgh>,
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

#ifndef ALPS_LATTICE_LATTICEGRAPHDESCRIPTOR_H
#define ALPS_LATTICE_LATTICEGRAPHDESCRIPTOR_H

#include <alps/config.h>
#include <alps/parameter.h>
#include <alps/lattice/disorder.h>
#include <alps/lattice/lattice.h>
#include <alps/lattice/latticegraph.h>
#include <alps/lattice/latticedescriptor.h>
#include <alps/lattice/hypercubic.h>
#include <alps/lattice/coordinatelattice.h>
#include <alps/lattice/coordinategraph.h>
#include <alps/utility/vectorio.hpp>

#include <iostream>

namespace alps {

class ALPS_DECL LatticeGraphDescriptor
  : public hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell>,std::vector<StringValue> >, std::vector<StringValue> >
{
public:
  typedef hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell>, std::vector<StringValue> >, std::vector<StringValue> > base_type;
  typedef lattice_traits<base_type>::unit_cell_type unit_cell_type;
  typedef lattice_traits<base_type>::offset_type offset_type;
  typedef lattice_traits<base_type>::extent_type extent_type;
  typedef lattice_traits<base_type>::cell_descriptor cell_descriptor;
  typedef lattice_traits<base_type>::vector_type vector_type;
  typedef lattice_traits<base_type>::basis_vector_iterator basis_vector_iterator;
  typedef lattice_traits<base_type>::cell_iterator cell_iterator; 
  typedef lattice_traits<base_type>::size_type size_type;
  typedef lattice_traits<base_type>::boundary_crossing_type boundary_crossing_type;

  LatticeGraphDescriptor() {}

  LatticeGraphDescriptor(const std::string& unitcell, const UnitCellMap&);
  
  LatticeGraphDescriptor(const XMLTag&, std::istream&, 
       const LatticeMap& = LatticeMap(), 
       const FiniteLatticeMap& = FiniteLatticeMap(), 
       const UnitCellMap& = UnitCellMap());

  void write_xml(oxstream&) const;
  const std::string& name() const { return name_;}
  void set_parameters(const Parameters&);
  const InhomogeneityDescriptor& inhomogeneity() const { return inhomogeneity_;}
  const DepletionDescriptor& depletion() const { return depletion_;}
private:
  std::string name_, lattice_name_, unitcell_name_;
  bool lattice_is_finite_;
  InhomogeneityDescriptor inhomogeneity_;
  DepletionDescriptor depletion_;
  FiniteLatticeDescriptor finitelattice_; 
  LatticeDescriptor lattice_; // for printing only
};

template<>
struct lattice_traits<LatticeGraphDescriptor>
{
  typedef LatticeGraphDescriptor::unit_cell_type unit_cell_type;
  typedef LatticeGraphDescriptor::cell_descriptor cell_descriptor;
  typedef LatticeGraphDescriptor::offset_type offset_type;
  typedef LatticeGraphDescriptor::extent_type extent_type;
  typedef LatticeGraphDescriptor::basis_vector_iterator basis_vector_iterator;
  typedef LatticeGraphDescriptor::cell_iterator cell_iterator;
  typedef LatticeGraphDescriptor::momentum_iterator momentum_iterator;
  typedef LatticeGraphDescriptor::size_type size_type;
  typedef LatticeGraphDescriptor::vector_type vector_type;
  typedef LatticeGraphDescriptor::boundary_crossing_type boundary_crossing_type;
};

typedef lattice_graph<LatticeGraphDescriptor,coordinate_graph_type> HypercubicLatticeGraphDescriptor;
typedef lattice_graph<hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > >, coordinate_graph_type> HypercubicLatticeGraph;

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<< (alps::oxstream& out, const alps::LatticeGraphDescriptor& l)
  {
    l.write_xml(out);
    return out;
  }

inline std::ostream& operator<< (std::ostream& out, const alps::LatticeGraphDescriptor& l)
  {
    alps::oxstream xml(out);
    xml << l;
    return out;
  }


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_LATTICE_LATTICEGRAPHDESCRIPTOR_H
