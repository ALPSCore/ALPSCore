/***************************************************************************
* ALPS++/lattice library
*
* lattice/unitcell.h     the unit cell of a lattice
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

#ifndef ALPS_LATTICE_UNITCELL_H
#define ALPS_LATTICE_UNITCELL_H

#include <alps/config.h>
#include <alps/vectormath.h>
#ifndef ALPS_WITHOUT_XML
# include <alps/parser/parser.h>
#endif
#include <alps/lattice/graph.h>
#include <alps/lattice/graphproperties.h>
#include <alps/lattice/dimensional_traits.h>
#include <boost/graph/adjacency_list.hpp>

namespace alps {

class EmptyUnitCell {
public:
  EmptyUnitCell(std::size_t d=0) : dim_(d) {}
  std::size_t dimension() const {return dim_;}
private:
  std::size_t dim_;	
};

inline dimensional_traits<EmptyUnitCell>::dimension_type
dimension(EmptyUnitCell c)
{
  return c.dimension();
}

class GraphUnitCell
{
public:
  typedef std::vector<int> offset_type;
  typedef detail::coordinate_type coordinate_type;
  typedef boost::adjacency_list<boost::vecS,boost::vecS,boost::directedS,
                                // vertex property
                                boost::property<coordinate_t,detail::coordinate_type,
				  boost::property<vertex_type_t,int> >,
				// edge property
				boost::property<target_offset_t,offset_type,
				  boost::property<source_offset_t,offset_type,
				    boost::property<edge_type_t,int > > >
				> graph_type;

  GraphUnitCell() {}
  GraphUnitCell(const EmptyUnitCell& e) : dim_(alps::dimension(e)) {}
#ifndef ALPS_WITHOUT_XML
  GraphUnitCell(const alps::XMLTag&, std::istream&);
#endif

  const GraphUnitCell& operator=(const EmptyUnitCell& e)
  {
    if (dim_==0) dim_=alps::dimension(e);
    return *this;
  }

#ifndef ALPS_WITHOUT_XML
  void write_xml(std::ostream&, const std::string& = "") const;
#endif

  graph_type& graph() { return graph_;}
  const graph_type& graph() const { return graph_;}
  std::size_t dimension() const { return dim_;}
  const std::string& name() const { return name_;}
  
private:	
  graph_type graph_;
  std::size_t dim_;
  std::string name_;
};

template<>
struct graph_traits<GraphUnitCell> {
  typedef GraphUnitCell::graph_type graph_type;
};

inline dimensional_traits<GraphUnitCell>::dimension_type
dimension(const GraphUnitCell& c)
{
  return c.dimension();
}

typedef std::map<std::string,GraphUnitCell> UnitCellMap;

} // end namespace alps

#ifndef ALPS_WITHOUT_XML

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& out, const alps::GraphUnitCell& u)
{
  u.write_xml(out);
  return out;	
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif

#endif // ALPS_LATTICE_UNITCELL_H
