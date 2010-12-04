/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/expression.h>
#include <alps/lattice/latticedescriptor.h>
#include <alps/lattice/lattice.h>

#ifndef ALPS_WITHOUT_XML

namespace alps {

LatticeDescriptor::LatticeDescriptor(const XMLTag& intag, std::istream& p)
  : dim_(0)
{
  XMLTag tag(intag);
  name_ = tag.attributes["name"];
  dim_ = tag.attributes["dimension"]=="" ? 0
           : boost::lexical_cast<uint32_t,std::string>(tag.attributes["dimension"]);
  if(tag.attributes["ref"]!="")
    boost::throw_exception(std::runtime_error("Illegal ref attribute in fully defined <LATTICE>"));
  if (tag.type !=XMLTag::SINGLE) while(true) {
    tag=parse_tag(p);
    if(tag.name=="/LATTICE")
      return;
    else if (tag.name=="PARAMETER") {
      lparms_[tag.attributes["name"]]=tag.attributes["default"];
      if (tag.type != XMLTag::SINGLE) {
        tag=parse_tag(p);
        if (tag.name!="/PARAMETER")
          boost::throw_exception(std::runtime_error("closing tag </PARAMETER> missing in <LATTICE> element"));
      }
    }
    else if (tag.name=="BASIS")  {
      if (tag.type==XMLTag::SINGLE)
        continue;
      while(true)  {
        tag=parse_tag(p);
        if(tag.name=="/BASIS")
          break;
        else if (tag.name=="VECTOR")  {
          if (tag.type==XMLTag::SINGLE)
            boost::throw_exception(std::runtime_error("coordinate contents expected in <VECTOR>"));
          add_basis_vector(read_vector<vector_type>(parse_content(p),dimension()));
          tag=parse_tag(p);
          if(tag.name!="/VECTOR")
              boost::throw_exception(std::runtime_error("invalid element <"+tag.name+
                        "> encountered in <VECTOR>"));
        }
        else
          boost::throw_exception(std::runtime_error("invalid element <" + tag.name + "> encountered in <BASIS>"));
        }
    }
    else if (tag.name=="RECIPROCALBASIS")  {
      if (tag.type==XMLTag::SINGLE)
        continue;
      while(true)  {
        tag=parse_tag(p);
        if(tag.name=="/RECIPROCALBASIS")
          break;
        else if (tag.name=="VECTOR")  {
          if (tag.type==XMLTag::SINGLE)
            boost::throw_exception(std::runtime_error("coordinate contents expected in <VECTOR>"));
          add_reciprocal_basis_vector(read_vector<vector_type>(parse_content(p),dimension()));
          tag=parse_tag(p);
          if(tag.name!="/VECTOR")
              boost::throw_exception(std::runtime_error("invalid element <"+tag.name+
                        "> encountered in <VECTOR>"));
        }
        else
          boost::throw_exception(std::runtime_error("invalid element <" + tag.name + "> encountered in <RECIPROCALBASIS>"));
        }
    }
    else
      boost::throw_exception(std::runtime_error("invalid tag <" + tag.name + "> encountered in <LATTICE>"));
  }
  if (!num_basis_vectors() && num_basis_vectors() != dimension())
    boost::throw_exception(std::runtime_error("incorrect number of basis vectors in <LATTICE>"));
}


void LatticeDescriptor::write_xml(oxstream& xml) const
{
  xml << start_tag("LATTICE");
  if (name() != "") xml << attribute("name", name());
  xml << attribute("dimension", dimension());
  for (Parameters::const_iterator it = lparms_.begin(); it != lparms_.end();
       ++it)
    xml << start_tag("PARAMETER") << attribute("name", it->key())
        << attribute("default", it->value()) << end_tag("PARAMETER");
  if (num_basis_vectors()) {
    xml << start_tag("BASIS");
    basis_vector_iterator v, v_end;
    for (boost::tie(v, v_end) = basis_vectors(); v != v_end; ++v)
      xml << start_tag("VECTOR") << no_linebreak << write_vector(*v)
          << end_tag("VECTOR");
    xml << end_tag("BASIS");
  }
  if (num_reciprocal_basis_vectors()) {
    xml << start_tag("RECIPROCALBASIS");
    basis_vector_iterator v, v_end;
    for (boost::tie(v, v_end) = reciprocal_basis_vectors(); v != v_end; ++v)
      xml << start_tag("VECTOR") << no_linebreak << write_vector(*v)
          << end_tag("VECTOR");
    xml << end_tag("RECIPROCALBASIS");
  }
  xml << end_tag("LATTICE");
}


FiniteLatticeDescriptor::FiniteLatticeDescriptor()
  : name_("open chain"),
    lattice_name_("open chain"),
    dim_(1)
{
  extent_.resize(dimension(),"L");
  bc_.resize(dimension(),"open");
}


FiniteLatticeDescriptor::FiniteLatticeDescriptor(const XMLTag& intag,
  std::istream& p, const LatticeMap& lm) : dim_(0)
{
  XMLTag tag(intag);
  name_ = tag.attributes["name"];
  dim_ = tag.attributes["dimension"]=="" ? 0
           : boost::lexical_cast<uint32_t,std::string>(tag.attributes["dimension"]);
  if(tag.attributes["ref"]!="")
    boost::throw_exception(std::runtime_error("Illegal ref attribute in fully defined <FINITELATTICE>"));

  if (tag.type == XMLTag::SINGLE)
    boost::throw_exception(std::runtime_error("missing <LATTICE> element in <FINITELATTICE>"));

  tag=parse_tag(p);
  if (tag.name!="LATTICE")
    boost::throw_exception(std::runtime_error("missing <LATTICE> element in <FINITELATTICE>"));

  lattice_name_=tag.attributes["ref"];
  if(lattice_name_!="")   {
    if (tag.type !=XMLTag::SINGLE) {
      tag=parse_tag(p);
      if(tag.name!="/LATTICE")
         boost::throw_exception(std::runtime_error("illegal contents in <LATTICE> reference tag"));
    }
    if(lm.find(lattice_name_)==lm.end())
      boost::throw_exception(std::runtime_error("unknown lattice: " + lattice_name_ + " in <FINITELATTICE>"));
    lattice_=lm.find(lattice_name_)->second;
  }
  else
    lattice_=LatticeDescriptor(tag,p);

  static_cast<base_type::parent_lattice_type&>(*this)
   = static_cast<const LatticeDescriptor::base_type&>(lattice_);

  if (dim_==0)
    dim_ = alps::dimension(lattice_);
  else if (alps::dimension(lattice_)!=0 && (alps::dimension(lattice_) !=dim_))
    boost::throw_exception(std::runtime_error("inconsistent lattice dimension between <LATTICE> and enclosing <FINITELATTICE>"));

  extent_.resize(dimension(),1);
  bc_.resize(dimension(),"open");
  while(true)  {
    tag=parse_tag(p);
    if(tag.name=="/FINITELATTICE") break;
    else if (tag.name=="PARAMETER") {
      flparms_[tag.attributes["name"]]=tag.attributes["default"];
      if (tag.type != XMLTag::SINGLE) {
        tag=parse_tag(p);
        if (tag.name!="/PARAMETER")
          boost::throw_exception(std::runtime_error("closing tag </PARAMETER> missing in <LATTICE> element"));
      }
    }
    else if (tag.name=="BOUNDARY")  {
      std::string bc = tag.attributes["type"];
      uint32_t dim=0;
          if (tag.attributes["dimension"]!="") {
        dim = boost::lexical_cast<uint32_t,std::string>(tag.attributes["dimension"]);
                if (dim==0 || dim>dimension())
          boost::throw_exception(std::runtime_error("incorrect dimension attribute in <BOUNDARY>"));
          }
      if (bc=="")
        boost::throw_exception(std::runtime_error("missing type attribute in <BOUNDARY>"));
      if (dim==0)
        std::fill(bc_.begin(),bc_.end(),bc);
      else
        bc_[dim-1]=bc;
    }
    else if(tag.name=="EXTENT")  {
      uint32_t dim=0;
          if (tag.attributes["dimension"]!="") {
            dim =  boost::lexical_cast<uint32_t,std::string>(tag.attributes["dimension"]);
                if (dim==0 || dim >dimension())
          boost::throw_exception(std::runtime_error("incorrect dimension attribute in <BOUNDARY>"));
          }
      std::string ex =  tag.attributes["size"];
      if (ex=="")
        boost::throw_exception(std::runtime_error("missing size attribute in <EXTENT>"));
      if (dim==0)
        std::fill(extent_.begin(),extent_.end(),ex);
      else
        extent_[dim-1]=ex;
    }
    else
      boost::throw_exception(std::runtime_error("invalid element <" + tag.name + "> encountered in <FINITELATTICE>"));
  }
  if (alps::dimension(extent_)!=dimension())
    boost::throw_exception(std::runtime_error("<EXTENT> element missing in <FINITELATTICE>"));
  if (alps::dimension(bc_)!=dimension())
    boost::throw_exception(std::runtime_error("<BOUNDARY> element missing in <FINITELATTICE>"));
}

void FiniteLatticeDescriptor::write_xml(oxstream& xml) const
{
  xml << start_tag("FINITELATTICE");
  if(name()!="")
    xml << attribute("name",  name());
  if (lattice_name_=="")
    xml << lattice_;
  else
    xml << start_tag("LATTICE") << attribute("ref", lattice_name_) << end_tag("LATTICE");
  for (Parameters::const_iterator it=flparms_.begin();it!=flparms_.end();++it)
    xml << start_tag("PARAMETER") << attribute("name", it->key())
        << attribute("default", it->value()) << end_tag("PARAMETER");
  for (unsigned int i=0;i<dimension();++i)
    xml << start_tag("EXTENT") << attribute("dimension", i+1) << attribute("size", extent_[i]) << end_tag();
  for (unsigned int i=0;i<dimension();++i)
    if (bc_[i] != "")
      xml  << start_tag("BOUNDARY") << attribute("dimension", i+1) << attribute("type", bc_[i]) << end_tag();
  xml << end_tag("FINITELATTICE");
}
#endif

void LatticeDescriptor::set_parameters(const Parameters& p)
{
  Parameters parms(lparms_);
  parms << p;
  base_type::set_parameters(parms);
}

void FiniteLatticeDescriptor::set_parameters(const Parameters& p)
{
  lattice_.set_parameters(p);
  static_cast<base_base_type&>(*this) = lattice_;
  Parameters parms(flparms_);
  parms << p;
  for (unsigned int i=0;i<bc_.size();++i) {
    if(bc_[i]!="")
      while (parms.defined(bc_[i]) && static_cast<std::string>(parms[bc_[i]]) != bc_[i])
        bc_[i] = static_cast<std::string>(parms[bc_[i]]);
    extent_[i] = static_cast<int>(alps::evaluate<double>(extent_[i], parms));
  }
}

void prevent_optimization() {}

} // end namespace alps
