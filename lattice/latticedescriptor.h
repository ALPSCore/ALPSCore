/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2006 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_LATTICE_LATTICEDESCRIPTOR_H
#define ALPS_LATTICE_LATTICEDESCRIPTOR_H

#include <alps/parameter.h>
#include <alps/parser/parser.h>
#include <alps/parser/xmlstream.h>
#include <alps/lattice/coordinatelattice.h>
#include <alps/lattice/hypercubic.h>

namespace alps {

 class ALPS_DECL LatticeDescriptor : public coordinate_lattice<simple_lattice<>,std::vector<alps::StringValue> >
{
public:
  typedef coordinate_lattice<simple_lattice<>, std::vector<alps::StringValue> >
    base_type;
  typedef lattice_traits<base_type>::unit_cell_type unit_cell_type;
  typedef lattice_traits<base_type>::offset_type offset_type;
  typedef lattice_traits<base_type>::cell_descriptor cell_descriptor;
  typedef lattice_traits<base_type>::vector_type vector_type;
  typedef lattice_traits<base_type>::basis_vector_iterator
    basis_vector_iterator;
  
  LatticeDescriptor() : base_type(), lparms_(), name_(), dim_(0) {}
  LatticeDescriptor(const std::string& name, std::size_t dim) :
    base_type(), lparms_(), name_(name), dim_(dim) {}
  LatticeDescriptor(const alps::XMLTag&, std::istream&);

  void write_xml(oxstream&) const;
  const std::string& name() const { return name_;}
  std::size_t dimension() const { return dim_;}

  void set_parameters(const alps::Parameters&);

  template<class T>
  void add_default_parameter(const std::string& name,
                             const T& value)
  { lparms_[name] = value; }

private:
  alps::Parameters lparms_;
  std::string name_;
  std::size_t dim_;
};

typedef std::map<std::string,LatticeDescriptor> LatticeMap;

class ALPS_DECL FiniteLatticeDescriptor : public hypercubic_lattice<coordinate_lattice<simple_lattice<>,std::vector<alps::StringValue> >, std::vector<alps::StringValue> >
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
  
  FiniteLatticeDescriptor();
  

  FiniteLatticeDescriptor(const alps::XMLTag&, std::istream&, 
                          const LatticeMap& = LatticeMap());

  void write_xml(oxstream&) const;

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

inline alps::oxstream& operator<<(alps::oxstream& out, const alps::LatticeDescriptor& l)
{
  l.write_xml(out);
  return out;
}

inline alps::oxstream& operator<<(alps::oxstream& out, const alps::FiniteLatticeDescriptor& l)
{
  l.write_xml(out);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::LatticeDescriptor& l)
{
  alps::oxstream xml(out);
  xml << l;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::FiniteLatticeDescriptor& l)
{
  alps::oxstream xml(out);
  xml << l;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_LATTICE_LATTICEDESCRIPTOR_H
