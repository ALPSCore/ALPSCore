/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_LATTICE_PARITY_H
#define ALPS_LATTICE_PARITY_H

#include <alps/config.h>
#include <alps/lattice/graphproperties.h>

#include <boost/graph/undirected_dfs.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/throw_exception.hpp>
#include <boost/vector_property_map.hpp>
#include <stdexcept>

namespace alps {

namespace parity {

typedef int8_t parity_type;
BOOST_STATIC_CONSTANT(parity_type, white = 0);
BOOST_STATIC_CONSTANT(parity_type, black = 1);
BOOST_STATIC_CONSTANT(parity_type, undefined = 2);

template<class Graph, class PropertyMap>
class ParityVisitor : public boost::dfs_visitor<>
{
public:
  typedef typename boost::graph_traits<Graph>::vertex_descriptor
    vertex_descriptor;
  typedef typename boost::graph_traits<Graph>::edge_descriptor
    edge_descriptor;

  ParityVisitor(PropertyMap& map, bool* check) :
    p_(white), map_(map), check_(check) { *check_ = true; }

  void discover_vertex(vertex_descriptor s, const Graph&)
  {
    flip();
    map_[s] = p_;
  }
  void back_edge(edge_descriptor e, const Graph& g) { check(e, g); }
  void finish_vertex(vertex_descriptor, const Graph&) { flip(); }

protected:
  void flip() { p_ = (p_ == white ? black : white); }
  void check(edge_descriptor e, const Graph& g) {
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
  typedef typename parity::ParityVisitor<Graph, Map> visitor_type;

  bool check = true;

  std::vector<boost::default_color_type> vcolor_map(boost::num_vertices(g));
  std::vector<boost::default_color_type> ecolor_map(boost::num_edges(g));
  boost::undirected_dfs(g, visitor_type(map, &check),
    boost::make_iterator_property_map(vcolor_map.begin(),
      boost::get(vertex_index_t(), g)),
    boost::make_iterator_property_map(ecolor_map.begin(),
      boost::get(edge_index_t(), g)));

  if (!check) {
    typename boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
      map[*vi]=parity::undefined;
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
  static bool set_parity(Graph& g)
  {
    typedef typename property_map<parity_t, Graph, int>::type map_type;
    map_type map = boost::get(parity_t(), g);
    return alps::set_parity(map, g);
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
