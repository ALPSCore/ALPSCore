/**************************************************************************
* ALPS++/model library
*
* model/library.C    a library for storing models
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/model/modellibrary.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/parser/parser.h>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {

void ModelLibrary::read_xml(const XMLTag& intag, std::istream& p)
{
  XMLTag tag(intag);
  if (tag.name !="MODELS")
    boost::throw_exception(std::runtime_error("<MODELS> tag needed at start of model library"));
  tag=parse_tag(p);
  while (tag.name!="/MODELS") {
     if (tag.name=="SITEBASIS")
      sitebases_[tag.attributes["name"]]=SiteBasisDescriptor<short>(tag,p);
     else if (tag.name=="BASIS")
      bases_[tag.attributes["name"]]=BasisDescriptor<short>(tag,p,sitebases_);
    else if (tag.name=="OPERATOR")
      operators_[tag.attributes["name"]]=OperatorDescriptor<short>(tag,p);
    else if (tag.name=="HAMILTONIAN")
      hamiltonians_[tag.attributes["name"]]=HamiltonianDescriptor<short>(tag,p,bases_);
    else
      boost::throw_exception(std::runtime_error("encountered unknown tag <" + tag.name+ "> while parsing <MODELS>"));
    tag=parse_tag(p);
  }
}

void ModelLibrary::write_xml(std::ostream& out) const
{
  out << "<MODELS>\n";
  for (SiteBasisDescriptorMap::const_iterator it=sitebases_.begin();it!=sitebases_.end();++it)
    out << it->second;
  for (BasisDescriptorMap::const_iterator it=bases_.begin();it!=bases_.end();++it)
    out << it->second;
  for (OperatorDescriptorMap::const_iterator it=operators_.begin();it!=operators_.end();++it)
    out << it->second;
  for (HamiltonianDescriptorMap::const_iterator it=hamiltonians_.begin();it!=hamiltonians_.end();++it)
    out << it->second;
  out << "</MODELS>\n";
}

bool ModelLibrary::has_basis(const std::string& name) const
{
  return (bases_.find(name)!=bases_.end());
}

bool ModelLibrary::has_hamiltonian(const std::string& name) const
{
  return (hamiltonians_.find(name)!=hamiltonians_.end());
}

bool ModelLibrary::has_site_basis(const std::string& name) const
{
  return (sitebases_.find(name)!=sitebases_.end());
}


bool ModelLibrary::has_operator(const std::string& name) const
{
  return (operators_.find(name)!=operators_.end());
}

void ModelLibrary::write_all_sets(std::ostream& os) const
{
  for (SiteBasisDescriptorMap::const_iterator it=sitebases_.begin(); it !=sitebases_.end();++it)
    os << "States of basis " << it->first << "=" << SiteBasisStates<short>(it->second);
}

const BasisDescriptor<short>& ModelLibrary::basis(const std::string& name) const
{
  if (!has_basis(name))
    boost::throw_exception(std::runtime_error("No basis named '" +name+"' found in model library"));
  return bases_.find(name)->second;
}

const HamiltonianDescriptor<short>& ModelLibrary::hamiltonian(const std::string& name) const
{
  if (!has_hamiltonian(name))
    boost::throw_exception(std::runtime_error("No Hamiltonian named '" +name+"' found in model library"));
  return hamiltonians_.find(name)->second;
}

const SiteBasisDescriptor<short>& ModelLibrary::site_basis(const std::string& name) const
{
  if (!has_site_basis(name))
    boost::throw_exception(std::runtime_error("No site basis named '" +name+"' found in model library"));
  return sitebases_.find(name)->second;
}

const OperatorDescriptor<short>& ModelLibrary::simple_operator(const std::string& name) const
{
  if (!has_operator(name))
    boost::throw_exception(std::runtime_error("No operator named '" +name+"' found in model library"));
  return operators_.find(name)->second;
}

} // namespace alps

#endif
