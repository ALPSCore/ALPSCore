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

#include <alps/lattice/latticegraphdescriptor.h>

#include <alps/lattice/latticegraph.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/lattice.h>

#include <iostream>

namespace alps {


LatticeGraphDescriptor::LatticeGraphDescriptor(const std::string& u, const UnitCellMap& um)
   : unitcell_name_(u), lattice_is_finite_(true)
{
  static_cast<base_type&>(*this) = FiniteLatticeDescriptor();
  
  if(um.find(unitcell_name_)==um.end())
    boost::throw_exception(std::runtime_error("unknown unit cell: " + unitcell_name_));
  unit_cell() = const_cast<UnitCellMap&>(um)[unitcell_name_];
  add_basis_vector(vector_type(1,"1"));
  add_reciprocal_basis_vector(vector_type(1,"1"));
}


LatticeGraphDescriptor::LatticeGraphDescriptor(const XMLTag& intag,
  std::istream& p, const LatticeMap& lm, const FiniteLatticeMap& flm,
  const UnitCellMap& um)
{
  XMLTag tag(intag);

  name_ = tag.attributes["name"];

  if (tag.type ==XMLTag::SINGLE) 
    boost::throw_exception(std::runtime_error("no lattice specified in <LATTICEGRAPH> element"));
  tag=parse_tag(p);
  if (tag.name=="LATTICE") {
    lattice_is_finite_=false;
    lattice_name_=tag.attributes["ref"];
    if(lattice_name_!="") {
      if (tag.type !=XMLTag::SINGLE) {
        tag=parse_tag(p);
        if(tag.name!="/LATTICE")
          boost::throw_exception(std::runtime_error("illegal contents in <LATTICE> reference tag"));
      }
      if(lm.find(lattice_name_)==lm.end())
        boost::throw_exception(std::runtime_error("unknown lattice: " + lattice_name_));
      lattice_ = lm.find(lattice_name_)->second;
    }
    else
      lattice_=LatticeDescriptor(tag,p);
  }
  else if(tag.name=="FINITELATTICE")
 {
    lattice_is_finite_=true;
    lattice_name_=tag.attributes["ref"];
    if(lattice_name_!="") {
      if (tag.type !=XMLTag::SINGLE) {
        tag=parse_tag(p);
        if(tag.name!="/FINITELATTICE")
          boost::throw_exception(std::runtime_error("illegal contents in <LATTICE> reference tag"));
      }
      if(flm.find(lattice_name_)==flm.end())
          boost::throw_exception(std::runtime_error("unknown lattice: " + lattice_name_));
      finitelattice_ = flm.find(lattice_name_)->second;
    }
    else
      finitelattice_=FiniteLatticeDescriptor(tag,p,lm);
    static_cast<base_type&>(*this) = finitelattice_;
  } 
  
  tag=parse_tag(p);
  if (tag.name!="UNITCELL")
    boost::throw_exception(std::runtime_error("<UNITCELL> element missing in <LATTICEGRAPH>"));
    
  unitcell_name_=tag.attributes["ref"];
  if(unitcell_name_!="") {
    if (tag.type !=XMLTag::SINGLE) {
      tag=parse_tag(p);
      if(tag.name!="/UNITCELL")
        boost::throw_exception(std::runtime_error("illegal contents in <UNITCELL> reference tag"));
    }
    if(um.find(unitcell_name_)==um.end())
      boost::throw_exception(std::runtime_error("unknown unit cell: " + unitcell_name_));
    unit_cell() = const_cast<UnitCellMap&>(um)[unitcell_name_];
  }
  else
    unit_cell() = GraphUnitCell(tag,p);

  tag=parse_tag(p);
  bool found_inhomogeneity = false;
  bool found_depletion = false;
  while (tag.name != "/LATTICEGRAPH") {
    if(tag.name=="INHOMOGENEOUS") {
      if (found_inhomogeneity)
        boost::throw_exception(std::runtime_error("duplicated <" + tag.name + "> tag in LATTICEGRAPH"));
      found_inhomogeneity = true;
      inhomogeneity_=InhomogeneityDescriptor(tag,p);
    } else if (tag.name=="DEPLETION") {
      if (found_depletion)
        boost::throw_exception(std::runtime_error("duplicated <" + tag.name + "> tag in LATTICEGRAPH"));
      found_depletion = true;
      depletion_=DepletionDescriptor(tag,p);
    } else {
      boost::throw_exception(std::runtime_error("illegal element <" + tag.name + "> in LATTICEGRAPH"));
    }
  }
}

void LatticeGraphDescriptor::set_parameters(const Parameters& p)
{
  if(lattice_is_finite_)
    finitelattice_.set_parameters(p);
  static_cast<base_type&>(*this) = finitelattice_;
  depletion_.set_parameters(p);
}

void LatticeGraphDescriptor::write_xml(oxstream& xml) const
{
  xml << start_tag("LATTICEGRAPH");
  if(name()!="")
    xml << attribute("name", name());
  if(lattice_is_finite_) {
    if (lattice_name_=="")
      xml << finitelattice_;
    else
      xml << start_tag("FINITELATTICE") << attribute("ref", lattice_name_) << end_tag("FINITELATTICE");
  }
  else {
    if (lattice_name_=="")
      xml << lattice_;
    else
      xml << start_tag("LATTICE") << attribute("ref", lattice_name_) << end_tag("LATTICE");
  }
  if (unitcell_name_=="")
    xml << unit_cell();
  else
    xml << start_tag("UNITCELL") << attribute("ref", unitcell_name_) << end_tag("UNITCELL");
  xml << inhomogeneity_ << depletion_;
  xml << end_tag("LATTICEGRAPH");
}

} // end namespace alps
