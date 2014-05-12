/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/model/modellibrary.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/parser/parser.h>
#include <alps/parser/xslt_path.h>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {

ModelLibrary::ModelLibrary(const Parameters& parms)
{
  std::string libname;
  if (parms.defined("MODEL_LIBRARY"))
    libname = static_cast<std::string>(parms["MODEL_LIBRARY"]);
  else
    libname = "models.xml";
    
  boost::filesystem::path p=search_xml_library_path(libname);
  
  std::ifstream libfile(p.string().c_str());
  if(!libfile)
    boost::throw_exception(std::runtime_error("Could not find model library file " + libname));
  read_xml(libfile);
}


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
      boost::throw_exception(std::runtime_error("Global operator descriptions were removed after ALPS 1.2"));
    else if (tag.name=="SITEOPERATOR")
      site_operators_[tag.attributes["name"]]=SiteOperator(tag,p);
    else if (tag.name=="BONDOPERATOR")
      bond_operators_[tag.attributes["name"]]=BondOperator(tag,p);
    else if (tag.name=="GLOBALOPERATOR")
      global_operators_[tag.attributes["name"]]=GlobalOperator(tag,p);
    else if (tag.name=="HAMILTONIAN")
      hamiltonians_[tag.attributes["name"]]=HamiltonianDescriptor<short>(tag,p,bases_,global_operators_);
    else
      boost::throw_exception(std::runtime_error("encountered unknown tag <" + tag.name+ "> while parsing <MODELS>"));
    tag=parse_tag(p);
  }
}

void ModelLibrary::write_xml(oxstream& out) const
{
  out << start_tag("MODELS");
  for (SiteBasisDescriptorMap::const_iterator it=sitebases_.begin();it!=sitebases_.end();++it)
    out << it->second;
  for (BasisDescriptorMap::const_iterator it=bases_.begin();it!=bases_.end();++it)
    out << it->second;
  for (SiteOperatorMap::const_iterator it=site_operators_.begin();it!=site_operators_.end();++it)
    out << it->second;
  for (BondOperatorMap::const_iterator it=bond_operators_.begin();it!=bond_operators_.end();++it)
    out << it->second;
  for (GlobalOperatorMap::const_iterator it=global_operators_.begin();it!=global_operators_.end();++it)
    out << it->second;
  for (HamiltonianDescriptorMap::const_iterator it=hamiltonians_.begin();it!=hamiltonians_.end();++it)
    out << it->second;
  out << end_tag("MODELS");
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

bool ModelLibrary::has_site_operator(const std::string& name) const
{
  return (site_operators_.find(name)!=site_operators_.end());
}

bool ModelLibrary::has_bond_operator(const std::string& name) const
{
  return (bond_operators_.find(name)!=bond_operators_.end());
}

bool ModelLibrary::has_global_operator(const std::string& name) const
{
  return (global_operators_.find(name)!=global_operators_.end());
}

const BasisDescriptor<short>& ModelLibrary::get_basis(const std::string& name) const
{
  if (!has_basis(name))
    boost::throw_exception(std::runtime_error("No basis named '" +name+"' found in model library"));
  return bases_.find(name)->second;
}

const HamiltonianDescriptor<short>& ModelLibrary::get_hamiltonian(const std::string& name) const
{
  if (!has_hamiltonian(name))
    boost::throw_exception(std::runtime_error("No Hamiltonian named '" +name+"' found in model library"));
  return hamiltonians_.find(name)->second;
}

HamiltonianDescriptor<short> ModelLibrary::get_hamiltonian(const std::string& name, Parameters const& parms, bool issymbolic) const
{
  Parameters p(parms); 
  alps::HamiltonianDescriptor<short> ham(get_hamiltonian(name));
  if (!issymbolic)
    p.copy_undefined(ham.default_parameters());
  ham.set_parameters(p);
  ham.substitute_operators(*this,issymbolic ? Parameters() : p);
  return ham;
}

const SiteBasisDescriptor<short>& ModelLibrary::get_site_basis(const std::string& name) const
{
  if (!has_site_basis(name))
    boost::throw_exception(std::runtime_error("No site basis named '" +name+"' found in model library"));
  return sitebases_.find(name)->second;
}

SiteOperator ModelLibrary::get_site_operator(const std::string& name,Parameters const& p) const
{
  if (!has_site_operator(name))
    boost::throw_exception(std::runtime_error("No site operator named '" +name+"' found in model library"));
  SiteOperator op(site_operators_.find(name)->second);
  op.substitute_operators(*this,p);
  return op;
}

BondOperator ModelLibrary::get_bond_operator(const std::string& name,Parameters const& p) const
{
  if (!has_bond_operator(name))
    boost::throw_exception(std::runtime_error("No bond operator named '" +name+"' found in model library"));
  BondOperator op(bond_operators_.find(name)->second);
  op.substitute_operators(*this,p);
  return op;
}

GlobalOperator ModelLibrary::get_global_operator(const std::string& name,Parameters const& p) const
{
  if (!has_global_operator(name))
    boost::throw_exception(std::runtime_error("No bond operator named '" +name+"' found in model library"));
  GlobalOperator op(global_operators_.find(name)->second);
  op.substitute_operators(*this,p);
  return op;
}

} // namespace alps

#endif
