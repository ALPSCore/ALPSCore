/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/lattice/latticelibrary.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/parser/parser.h>
#include <alps/parser/xslt_path.h>

namespace alps {

LatticeLibrary::LatticeLibrary(const Parameters& parms)
{
  std::string libname;
  if (parms.defined("LATTICE_LIBRARY"))
    libname = static_cast<std::string>(parms["LATTICE_LIBRARY"]);
  else
    libname = "lattices.xml";
  
  boost::filesystem::path p=search_xml_library_path(libname);
  
  std::ifstream libfile(p.string().c_str());
  if(!libfile)
    boost::throw_exception(std::runtime_error("Could not find lattice library file " + libname));
  read_xml(libfile);
}

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
      read_graph_xml(tag,p,graphs_[tag.attributes["name"]]);
    }
    else
      boost::throw_exception(std::runtime_error("encountered unknown tag <" + tag.name+ "> while parsing <LATTICES>"));
  }
}

void LatticeLibrary::write_xml(oxstream& out) const
{
  out << start_tag("LATTICES");
  for (LatticeMap::const_iterator it=lattices_.begin();it!=lattices_.end();++it)
    out << it->second;
  for (FiniteLatticeMap::const_iterator it=finitelattices_.begin();it!=finitelattices_.end();++it)
    out << it->second;
  for (UnitCellMap::const_iterator it=unitcells_.begin();it!=unitcells_.end();++it)
    out << it->second;
  for (LatticeGraphMap::const_iterator it=latticegraphs_.begin();it!=latticegraphs_.end();++it)
    out << it->second;
  for (GraphMap::const_iterator it=graphs_.begin();it!=graphs_.end();++it)
    write_graph_xml(out,it->second,it->first);
  out << end_tag("LATTICES");
}

bool LatticeLibrary::has_graph(const std::string& name) const
{
  return (graphs_.find(name)!=graphs_.end());
}


hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > > LatticeLibrary::lattice(const std::string& name) const
  {
    return hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > >(lattice_descriptor(name));
  }

bool LatticeLibrary::has_lattice(const std::string& name) const
{
  return (latticegraphs_.find(name)!=latticegraphs_.end());
}

bool LatticeLibrary::has_unitcell(const std::string& name) const
{
  return (unitcells_.find(name)!=unitcells_.end());
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
#ifndef __FCC_VERSION
  for (LatticeGraphMap::const_iterator it=latticegraphs_.begin();
       it !=latticegraphs_.end();++it)
    graphs_["Graph created from " + it->first] =
      detail::graph_wrap(HypercubicLatticeGraph(it->second));
#else
  boost::throw_exception(std::runtime_error("make_all_graphs() not implemented for Fujitsu C++ compiler"));
#endif
}

} // end namespace alps

#endif
