/**************************************************************************
* ALPS++/lattice library
*
* lattice/latticegraph.C    the lattice graph class
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

#include <alps/lattice/latticegraphdescriptor.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/lattice/latticegraph.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/lattice.h>
#include <alps/vectorio.h>

#include <iostream>

namespace alps {

LatticeGraphDescriptor::LatticeGraphDescriptor(const XMLTag& intag, std::istream& p,
                          const LatticeMap& lm, const FiniteLatticeMap& flm, const UnitCellMap& um)
{
  XMLTag tag(intag);

  for (XMLTag::AttributeMap::const_iterator it=tag.attributes.begin();it!=tag.attributes.end();++it)
  {
    if(it->first=="name")
      name_=it->second;
    else
      boost::throw_exception(std::runtime_error("illegal attribute " + it->first + " in <LATTICEGRAPH> element"));
  }      
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
    unit_cell_=const_cast<UnitCellMap&>(um)[unitcell_name_];
  }
  else
    unit_cell_=GraphUnitCell(tag,p);

  tag=parse_tag(p);
  if(tag.name!="/LATTICEGRAPH")
    boost::throw_exception(std::runtime_error("illegal element <" + tag.name + "> in LATTICEGRAPH"));
}

void LatticeGraphDescriptor::set_parameters(const Parameters& p)
{
  if(lattice_is_finite_)
    finitelattice_.set_parameters(p);
  static_cast<base_type&>(*this) = finitelattice_;
}

void LatticeGraphDescriptor::write_xml(std::ostream& xml, const std::string& prefix) const
{
  xml << prefix << "<LATTICEGRAPH";
  if(name()!="")
    xml << " name=\"" << name() << "\"";
  xml << ">\n";
  if(lattice_is_finite_)
  {
    if (lattice_name_=="")
      finitelattice_.write_xml(xml, prefix + "  ");
    else
      xml << prefix << "  <FINITELATTICE ref=\"" << lattice_name_ << "\"/>\n";
  }
  else
  {
    if (lattice_name_=="")
      lattice_.write_xml(xml, prefix + "  ");
    else
      xml << prefix << "  <FINITELATTICE ref=\"" << lattice_name_ << "\"/>\n";
  }
  if (unitcell_name_=="")
    unit_cell_.write_xml(xml, prefix + "  ");
  else
    xml << prefix << "  <UNITCELL ref=\"" << unitcell_name_ << "\"/>\n";
  xml << prefix << "</LATTICEGRAPH>\n";
}

} // end namespace alps

#endif
