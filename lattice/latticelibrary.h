/***************************************************************************
* ALPS++/lattice library
*
* lattice/library.h    the lattice graph class
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

#ifndef ALPS_LATTICE_LIBRARY_H
#define ALPS_LATTICE_LIBRARY_H

#include <alps/config.h>

#ifndef ALPS_WITHOUT_XML

#include <alps/lattice/latticegraph.h>
#include <alps/lattice/latticegraphdescriptor.h>
#include <alps/lattice/latticedescriptor.h>
#include <alps/lattice/graph.h>

#include <fstream>

namespace alps {

class LatticeLibrary
{
public:
  LatticeLibrary() {};
  LatticeLibrary(std::istream& in) { read_xml(in);}
  LatticeLibrary(const alps::XMLTag& tag, std::istream& p) {read_xml(tag,p);}

  void read_xml(std::istream& in);
  void read_xml(const alps::XMLTag& tag, std::istream& p);

  void write_xml(std::ostream&) const;
  
  bool has_graph(const std::string& name) const;
  bool has_lattice(const std::string& name) const;
  
  const LatticeGraphDescriptor& lattice_descriptor(const std::string& name) const;
  hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > > lattice(const std::string& name) const
  {
    return hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > >(lattice_descriptor(name));
  }
  const coordinate_graph_type& graph(const std::string& name) const;
  
  template <class G>
  bool get_graph(G& graph,const std::string& name) const;
  
  void make_all_graphs();

private:
  typedef std::map<std::string,LatticeGraphDescriptor> LatticeGraphMap;
  typedef std::map<std::string,alps::coordinate_graph_type> GraphMap;

  LatticeMap lattices_;
  FiniteLatticeMap finitelattices_;
  UnitCellMap unitcells_;
  LatticeGraphMap latticegraphs_;
  GraphMap graphs_;
};

template <class G=coordinate_graph_type>
class graph_factory : public LatticeLibrary
{
public:
  typedef G graph_type;
  graph_factory() : g_(0), to_delete_(false) {}
  graph_factory(std::istream& in) : LatticeLibrary(in), g_(0), to_delete_(false) {}
  graph_factory(std::istream& in, const alps::Parameters& parm) 
    : LatticeLibrary(in), g_(0), to_delete_(false) { make_graph(parm);}
  graph_factory(const alps::Parameters& parm);
  ~graph_factory() { if (to_delete_) delete g_;}

  void make_graph(const alps::Parameters& p);
  graph_type& graph()
  {
    if (g_==0) boost::throw_exception(std::runtime_error("no graph created in graph_factory"));
    return *g_;
  }
  const graph_type& graph() const
  {
    if (g_==0) boost::throw_exception(std::runtime_error("no graph created in graph_factory"));
    return *g_;
  }

private:
  typedef lattice_graph<hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > >,graph_type> lattice_type;
  graph_type* g_;
  bool to_delete_;
  lattice_type l_;
};

template <class G>
inline bool LatticeLibrary::get_graph(G& g, const std::string& name) const
{
  if (!has_graph(name))
    return false;
  else
  {
    copy_graph(const_cast<GraphMap&>(graphs_)[name],g);
    return true;
  }
}

template <class G>
inline alps::graph_factory<G>::graph_factory(const alps::Parameters& parms)
 : g_(0), to_delete_(false)
{
  std::string libname;
  if (parms.defined("LATTICE_LIBRARY"))
    libname = static_cast<std::string>(parms["LATTICE_LIBRARY"]);
  else
    libname = "lattices.xml";
  std::ifstream libfile(libname.c_str());
  if(!libfile)
    boost::throw_exception(std::runtime_error("Could not find lattice library file " + libname));
  read_xml(libfile);
  make_graph(parms);
}

template <class G>
inline void
alps::graph_factory<G>::make_graph(const alps::Parameters& parms)
{
  std::string name;
  bool have_graph=false;
  bool have_lattice=false;
  
  if (have_graph = parms.defined("GRAPH"))
    name = static_cast<std::string>(parms["GRAPH"]);
  if (have_lattice = parms.defined("LATTICE"))
    name = static_cast<std::string>(parms["LATTICE"]);
  if (have_lattice && have_graph)
    boost::throw_exception(std::runtime_error("both GRAPH and LATTICE were specified"));
  if (have_lattice && has_lattice(name)) {
    LatticeGraphDescriptor desc(lattice_descriptor(name));
    desc.set_parameters(parms);
    l_ = lattice_type(desc);
    if (to_delete_)
      delete g_;
    g_ = &(l_.graph());
    to_delete_=false;
  }
  else if ((have_lattice || have_graph) && has_graph(name)) {
    if (to_delete_)
      delete g_;
    g_= new graph_type();
    get_graph(*g_,name);
    to_delete_=true;
  }
  else
    boost::throw_exception(std::runtime_error("could not find graph/lattice specified in parameters"));
}

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& os, const alps::LatticeLibrary& l)
{
  l.write_xml(os);
  return os;
}

inline std::istream& operator>>(std::istream& is, alps::LatticeLibrary& l)
{
  l.read_xml(is);
  return is;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif

#endif // ALPS_LATTICE_LIBRARY_H
