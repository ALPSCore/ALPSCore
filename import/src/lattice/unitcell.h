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

#ifndef ALPS_LATTICE_UNITCELL_H
#define ALPS_LATTICE_UNITCELL_H

#include <alps/config.h>
#include <alps/parser/parser.h>
#include <alps/parser/xmlstream.h>
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
dimension(const EmptyUnitCell& c)
{
  return c.dimension();
}

class ALPS_DECL GraphUnitCell
{
public:
  typedef std::vector<int> offset_type;
  typedef ::alps::coordinate_type coordinate_type;
  typedef boost::adjacency_list<boost::vecS,boost::vecS,boost::directedS,
                                // vertex property
                                boost::property<coordinate_t,coordinate_type,
                                  boost::property<vertex_type_t,type_type> >,
                                // edge property
                                boost::property<target_offset_t,offset_type,
                                  boost::property<source_offset_t,offset_type,
                                    boost::property<edge_type_t,type_type,
                                      boost::property<bond_vector_t,coordinate_type> > > >
                                > graph_type;

  GraphUnitCell();
  GraphUnitCell(const EmptyUnitCell& e);
  GraphUnitCell(const XMLTag&, std::istream&);
  GraphUnitCell(const std::string& name, std::size_t dim);

  const GraphUnitCell& operator=(const EmptyUnitCell& e);

  void write_xml(oxstream&) const;
  graph_type& graph() { return graph_;}
  const graph_type& graph() const { return graph_;}
  std::size_t dimension() const { return dim_;}
  const std::string& name() const { return name_;}

  std::size_t add_vertex(int type, const coordinate_type& coord);
  std::size_t add_edge(int type, uint32_t si, const offset_type& so,
                       uint32_t ti, const offset_type& to);

private:        
  void update_bond_vectors();
  graph_type graph_;
  std::size_t dim_;
  std::string name_;
};

inline dimensional_traits<GraphUnitCell>::dimension_type
dimension(const GraphUnitCell& c)
{ return c.dimension(); }


namespace graph{

inline GraphUnitCell::graph_type&
graph(GraphUnitCell& c) { return c.graph(); }

inline const GraphUnitCell::graph_type&
graph(const GraphUnitCell& c) { return c.graph(); }

} // end namespace graph

typedef std::map<std::string,GraphUnitCell> UnitCellMap;

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& out, const alps::GraphUnitCell& u)
{
  u.write_xml(out);
  return out;        
}

inline std::ostream& operator<<(std::ostream& out, const alps::GraphUnitCell& u)
{
  oxstream xml(out);
  xml << u;
  return out;        
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_LATTICE_UNITCELL_H
