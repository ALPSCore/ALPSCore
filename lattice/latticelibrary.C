/**************************************************************************
* ALPS++/lattice library
*
* lattice/library.C    a library for storing lattices
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/lattice/latticelibrary.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/parser/parser.h>

namespace alps {

void LatticeLibrary::read_xml(std::istream& in)
{
  XMLTag tag=parse_tag(in);
  read_xml(tag,in);
}

void LatticeLibrary::read_xml(const XMLTag& intag, std::istream& p)
{
  XMLTag tag(intag);
  if (tag.name !="LATTICES")
    boost::throw_exception(std::runtime_error("<LATTICES> tag needed at start of lattice library"));
  while (true)
  {
    XMLTag tag=parse_tag(p);
    if (tag.name=="/LATTICES")
      break;
    if (tag.name=="LATTICE")
      lattices_[tag.attributes["name"]]=LatticeDescriptor(tag,p);
    else if (tag.name=="FINITELATTICE")
      finitelattices_[tag.attributes["name"]]=FiniteLatticeDescriptor(tag,p,lattices_);
    else if (tag.name=="UNITCELL")
      unitcells_[tag.attributes["name"]]=GraphUnitCell(tag,p);
    else if (tag.name=="LATTICEGRAPH")
      latticegraphs_[tag.attributes["name"]]=
        LatticeGraphDescriptor(tag,p,lattices_,finitelattices_,unitcells_);
    else if (tag.name=="GRAPH") {
      graphs_[tag.attributes["name"]]=coordinate_graph_type();
      alps::read_graph_xml(tag,p,graphs_[tag.attributes["name"]]);
    }
    else
      boost::throw_exception(std::runtime_error("encountered unknown tag <" + tag.name+ "> while parsing <LATTICES>"));
  }
}

void LatticeLibrary::write_xml(std::ostream& out) const
{
  out << "<LATTICES>\n";
  for (LatticeMap::const_iterator it=lattices_.begin();it!=lattices_.end();++it)
    out << it->second;
  for (FiniteLatticeMap::const_iterator it=finitelattices_.begin();it!=finitelattices_.end();++it)
    out << it->second;
  for (UnitCellMap::const_iterator it=unitcells_.begin();it!=unitcells_.end();++it)
    out << it->second;
  for (LatticeGraphMap::const_iterator it=latticegraphs_.begin();it!=latticegraphs_.end();++it)
    out << it->second;
  for (GraphMap::const_iterator it=graphs_.begin();it!=graphs_.end();++it)
    alps::write_graph_xml(out,it->second,it->first);
  out << "</LATTICES>\n";
}

bool LatticeLibrary::has_graph(const std::string& name) const
{
  return (graphs_.find(name)!=graphs_.end());
}

bool LatticeLibrary::has_lattice(const std::string& name) const
{
  return (latticegraphs_.find(name)!=latticegraphs_.end());
}

const LatticeGraphDescriptor& LatticeLibrary::lattice_descriptor(const std::string& name) const
{
  if (!has_lattice(name))
    boost::throw_exception(std::runtime_error("No lattice named '" +name+"' found in lattice library"));
  return latticegraphs_.find(name)->second;
}

const coordinate_graph_type& LatticeLibrary::graph(const std::string& name) const
{
  if (!has_graph(name))
    boost::throw_exception(std::runtime_error("No graph named '" +name+"' found in lattice library"));
  return graphs_.find(name)->second;
}

void LatticeLibrary::make_all_graphs()
{
  for (LatticeGraphMap::const_iterator it=latticegraphs_.begin(); it !=latticegraphs_.end();++it)
    graphs_["Graph created from " + it->first]=alps::graph(HypercubicLatticeGraph(it->second));
}

} // end namespace alps

#endif
