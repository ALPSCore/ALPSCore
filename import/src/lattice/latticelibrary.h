/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_LIBRARY_H
#define ALPS_LATTICE_LIBRARY_H

#include <alps/config.h>
#include <alps/lattice/latticegraph.h>
#include <alps/lattice/latticegraphdescriptor.h>
#include <alps/lattice/latticedescriptor.h>
#include <alps/lattice/graph.h>
#include <alps/parser/xmlstream.h>

#include <fstream>

namespace alps {

class ALPS_DECL LatticeLibrary
{
public:
  typedef hypercubic_lattice<coordinate_lattice<simple_lattice<GraphUnitCell> > > lattice_type;
  
  LatticeLibrary() {};
  LatticeLibrary(std::istream& in) { read_xml(in);}
  LatticeLibrary(const XMLTag& tag, std::istream& p) {read_xml(tag,p);}
  LatticeLibrary(const Parameters& p);
  void read_xml(std::istream& in);
  void read_xml(const XMLTag& tag, std::istream& p);

  void write_xml(oxstream&) const;
  
  bool has_graph(const std::string& name) const;
  bool has_lattice(const std::string& name) const;
  bool has_unitcell(const std::string& name) const;
  
  const LatticeGraphDescriptor& lattice_descriptor(const std::string& name) const;
  lattice_type lattice(const std::string& name) const;
  const coordinate_graph_type& graph(const std::string& name) const;
  
  template <class G>
  bool get_graph(G& graph,const std::string& name) const;
  
  void make_all_graphs();

protected:
  typedef std::map<std::string,LatticeGraphDescriptor> LatticeGraphMap;
  typedef std::map<std::string,coordinate_graph_type> GraphMap;

  LatticeMap lattices_;
  FiniteLatticeMap finitelattices_;
  UnitCellMap unitcells_;
  LatticeGraphMap latticegraphs_;
  GraphMap graphs_;
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

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& xml, const alps::LatticeLibrary& l)
{
  l.write_xml(xml);
  return xml;
}

inline std::ostream& operator<<(std::ostream& os, const alps::LatticeLibrary& l)
{
  oxstream xml(os);
  xml << l;
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

#endif // ALPS_LATTICE_LIBRARY_H
