/***************************************************************************
* ALPS++/lattice library
*
* lattice/latticedescriptor.h    the lattice descriptor class
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

#ifndef ALPS_LATTICE_LATTICEDESCRIPTOR_H
#define ALPS_LATTICE_LATTICEDESCRIPTOR_H

#include <alps/config.h>

#include <alps/parameters.h>
#ifndef ALPS_WITHOUT_XML
# include <alps/parser/parser.h>
#endif
#include <alps/lattice/coordinatelattice.h>
#include <alps/lattice/hypercubic.h>

namespace alps {

class LatticeDescriptor : public coordinate_lattice<simple_lattice<>,std::vector<alps::StringValue> >
{
public:
  typedef coordinate_lattice<simple_lattice<>,std::vector<alps::StringValue> > base_type;
  typedef lattice_traits<base_type>::unit_cell_type unit_cell_type;
  typedef lattice_traits<base_type>::offset_type offset_type;
  typedef lattice_traits<base_type>::cell_descriptor cell_descriptor;
  typedef lattice_traits<base_type>::vector_type vector_type;
  typedef lattice_traits<base_type>::basis_vector_iterator basis_vector_iterator;
  
  LatticeDescriptor() : dim_(0) {}
#ifndef ALPS_WITHOUT_XML
  LatticeDescriptor(const alps::XMLTag&, std::istream&);
#endif

  void write_xml(std::ostream&, const std::string& = "") const;
  const std::string& name() const { return name_;}
  std::size_t dimension() const { return dim_;}

  void set_parameters(const alps::Parameters&);
private:
  alps::Parameters lparms_;
  std::string name_;
  std::size_t dim_;
};

typedef std::map<std::string,LatticeDescriptor> LatticeMap;

class FiniteLatticeDescriptor : public hypercubic_lattice<coordinate_lattice<simple_lattice<>,std::vector<alps::StringValue> >, std::vector<alps::StringValue> >
{
public:
  typedef hypercubic_lattice<coordinate_lattice<simple_lattice<>,std::vector<alps::StringValue> > > base_type;
  typedef coordinate_lattice<simple_lattice<>,std::vector<alps::StringValue> > base_base_type;
  typedef lattice_traits<base_type>::unit_cell_type unit_cell_type;
  typedef lattice_traits<base_type>::offset_type offset_type;
  typedef lattice_traits<base_type>::cell_descriptor cell_descriptor;
  typedef lattice_traits<base_type>::vector_type vector_type;
  typedef lattice_traits<base_type>::basis_vector_iterator basis_vector_iterator;
  typedef lattice_traits<base_type>::cell_iterator cell_iterator;
  typedef lattice_traits<base_type>::size_type size_type;
  
  FiniteLatticeDescriptor() : dim_(0) {}
  
#ifndef ALPS_WITHOUT_XML
  FiniteLatticeDescriptor(const alps::XMLTag&, std::istream&, 
                          const LatticeMap& = LatticeMap());
#endif

  void write_xml(std::ostream&, const std::string& n= "") const;

  const std::string& name() const { return name_;}
  void set_parameters(const alps::Parameters&);
  std::size_t dimension() const { return dim_;}

private:
  std::string name_;
  std::string lattice_name_;
  std::size_t dim_;
  alps::Parameters flparms_;

  LatticeDescriptor lattice_; // for printing only
};

inline dimensional_traits<LatticeDescriptor>::dimension_type
dimension(const LatticeDescriptor& c)
{
  return c.dimension();
}

inline dimensional_traits<FiniteLatticeDescriptor>::dimension_type
dimension(const FiniteLatticeDescriptor& c)
{
  return c.dimension();
}

typedef std::map<std::string,FiniteLatticeDescriptor> FiniteLatticeMap;

} // end namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& out, const alps::LatticeDescriptor& l)
{
  l.write_xml(out);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::FiniteLatticeDescriptor& l)
{
  l.write_xml(out);
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_LATTICE_LATTICEDESCRIPTOR_H
