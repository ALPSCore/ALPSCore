/**************************************************************************
* ALPS++/model library
*
* model/library.C    a library for storing models
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@comp-phys.org>
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
**************************************************************************/

#include <alps/model/modellibrary.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/parser/parser.h>
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
  std::ifstream libfile(libname.c_str());
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
      operators_[tag.attributes["name"]]=OperatorDescriptor<short>(tag,p);
    else if (tag.name=="HAMILTONIAN")
      hamiltonians_[tag.attributes["name"]]=HamiltonianDescriptor<short>(tag,p,bases_);
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
  for (OperatorDescriptorMap::const_iterator it=operators_.begin();it!=operators_.end();++it)
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


bool ModelLibrary::has_operator(const std::string& name) const
{
  return (operators_.find(name)!=operators_.end());
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
