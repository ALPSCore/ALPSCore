/***************************************************************************
* ALPS++/lattice library
*
* lattice/parity.h   setting parity for bipartite graphs
*
* $Id$
*
* Copyright (C) 2001-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_LATTICE_PARITY_H
#define ALPS_LATTICE_PARITY_H

#include <alps/config.h>
#include <alps/lattice/graphproperties.h>

#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {

namespace parity {

typedef int8_t parity_type;
BOOST_STATIC_CONSTANT(parity_type, white = 0);
BOOST_STATIC_CONSTANT(parity_type, black = 1);
BOOST_STATIC_CONSTANT(parity_type, undefined = 2);

template<class Graph, class PropertyMap>
class ParityVisitor
{
public:
  typedef typename boost::graph_traits<Graph>::edge_descriptor
    edge_descriptor;
  typedef typename boost::graph_traits<Graph>::vertex_descriptor
    vertex_descriptor;

  // constructor
  ParityVisitor(PropertyMap& map, bool* check) :
    p_(white), map_(map), check_(check) { *check_ = true; }

  // callback member functions
  void initialize_vertex(vertex_descriptor s, const Graph&) {
    map_[s]=undefined;
  }
  void start_vertex(vertex_descriptor, const Graph&) {}
  void discover_vertex(vertex_descriptor s, const Graph&) {
    flip();
    map_[s]=p_;
  }
  void examine_edge(edge_descriptor, const Graph&) {}
  void tree_edge(edge_descriptor, const Graph&) {}
  void back_edge(edge_descriptor e, const Graph& g) { check(e, g); }
  void forward_or_cross_edge(edge_descriptor e, const Graph& g) {
    check(e, g);
  }
  void finish_vertex(vertex_descriptor, const Graph&) { flip(); }

protected:
  ParityVisitor();

  void flip() { p_ = (p_ == white ? black : white); }
  void check(edge_descriptor e, const Graph& g) {
    if (map_[boost::source(e, g)] == undefined ||
 	map_[boost::target(e, g)] == undefined) {
      boost::throw_exception(std::runtime_error("unvisited vertex found"));
    }
    if (map_[boost::source(e, g)] == map_[boost::target(e, g)]) 
      *check_ = false;
  }

private:
  parity_type p_;
  PropertyMap map_;
  bool* check_;
};

} // end namespace parity

template <class Graph, class Map>
bool set_parity(Map map, const Graph& g)
{
  typedef typename boost::graph_traits<Graph>::vertex_iterator
    vertex_iterator;
  typedef typename parity::ParityVisitor<Graph, Map> visitor_type;
  bool check = true;
  visitor_type v(map, &check);
  boost::depth_first_search(g, boost::visitor(v));
  if (!check) {
    for (vertex_iterator itr = boost::vertices(g).first;
	  itr != boost::vertices(g).second; ++itr) {
      boost::put(map, *itr, parity::undefined);
    }
  }
  return check;
}

namespace parity {

template<bool HasParity>
struct helper
{
  template<class Graph>
  static bool set_parity(Graph&) { return false; }
};

template<>
struct helper<true>
{
  template<class Graph>
  static bool set_parity(Graph& g) {
    typedef typename property_map<parity_t, Graph, int>::type map_type;
    map_type map = boost::get(parity_t(), g);
    return alps::set_parity(map,g);
  }
};

} // end namespace parity


template<class Graph>
bool set_parity(Graph& g)
{
  return parity::helper<has_property<parity_t, Graph>::vertex_property>
    ::set_parity(g);
}

} // end namespace alps

#endif // ALPS_LATTICE_PARITY_H
