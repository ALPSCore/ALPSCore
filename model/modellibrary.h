/***************************************************************************
* ALPS++/model library
*
* model/library.h    the model graph class
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_MODEL_LIBRARY_H
#define ALPS_MODEL_LIBRARY_H

#include <alps/config.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/model/sitebasis.h>
#include <alps/model/operator.h>
#include <alps/model/basis.h>
#include <alps/parser/parser.h>

#include <string>

namespace alps {

class ModelLibrary
{
public:
  ModelLibrary() {};
  ModelLibrary(std::istream& in) { read_xml(in);}
  ModelLibrary(const XMLTag& tag, std::istream& p) {read_xml(tag,p);}

  void read_xml(std::istream& in) { read_xml(parse_tag(in),in);}
  void read_xml(const XMLTag& tag, std::istream& p);
  void write_xml(std::ostream&) const;
  
  bool has_basis(const std::string& name) const;
  bool has_site_basis(const std::string& name) const;
  bool has_operator(const std::string& name) const;
  bool has_hamiltonian(const std::string& name) const;
  
  const SiteBasisDescriptor<short>& site_basis(const std::string& name) const;
  const BasisDescriptor<short>& basis(const std::string& name) const;
  const OperatorDescriptor<short>& simple_operator(const std::string& name) const;
  const HamiltonianDescriptor<short>& hamiltonian(const std::string& name) const;
  
  SiteBasisStates<short> site_states(const std::string& name) const 
  { return SiteBasisStates<short>(site_basis(name));}

  void write_all_sets(std::ostream& os) const;

private:
  typedef std::map<std::string,SiteBasisDescriptor<short> > SiteBasisDescriptorMap;
  typedef std::map<std::string,BasisDescriptor<short> > BasisDescriptorMap;
  typedef std::map<std::string,OperatorDescriptor<short> > OperatorDescriptorMap;
  typedef std::map<std::string,HamiltonianDescriptor<short> > HamiltonianDescriptorMap;

  SiteBasisDescriptorMap sitebases_;
  BasisDescriptorMap bases_;
  OperatorDescriptorMap operators_;
  HamiltonianDescriptorMap hamiltonians_;
};

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& os, const alps::ModelLibrary& l)
{
  l.write_xml(os);
  return os;
}

inline std::istream& operator>>(std::istream& is, alps::ModelLibrary& l)
{
  l.read_xml(is);
  return is;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif

#endif // ALPS_MODEL_LIBRARY_H
