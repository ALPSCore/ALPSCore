/***************************************************************************
* ALPS++/lattice library
*
* lattice/graph_traits.h     default graph traits
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

#ifndef ALPS_LATTICE_GRAPH_TRAITS_H
#define ALPS_LATTICE_GRAPH_TRAITS_H

namespace alps {

template <class G>
struct graph_traits : public boost::graph_traits<G>
{
  typedef G graph_type;
  typedef typename boost::graph_traits<graph_type>::vertex_iterator site_iterator;
  typedef typename boost::graph_traits<graph_type>::edge_iterator bond_iterator;
  typedef typename boost::graph_traits<graph_type>::out_edge_iterator neighbor_bond_iterator;
  typedef typename boost::graph_traits<graph_type>::edge_descriptor bond_descriptor;
  typedef typename boost::graph_traits<graph_type>::vertex_descriptor site_descriptor;
  typedef typename boost::graph_traits<graph_type>::vertices_size_type sites_size_type;
  typedef typename boost::graph_traits<graph_type>::edges_size_type bonds_size_type;
  typedef typename boost::graph_traits<graph_type>::degree_size_type neighbors_size_type;
  typedef typename boost::graph_traits<graph_type>::adjacency_iterator neighbor_iterator;
};

template <class G> 
std::pair<typename graph_traits<G>::site_iterator,
          typename graph_traits<G>::site_iterator> sites(const G& g) {
  return boost::vertices(g);
}

template <class G> 
typename graph_traits<G>::sites_size_type num_sites (const G& g) {
  return boost::num_vertices(g);
}

template <class G> 
std::pair<typename graph_traits<G>::bond_iterator,
          typename graph_traits<G>::bond_iterator> bonds(const G& g) {
  return boost::edges(g);
}

template <class G> 
typename graph_traits<G>::bonds_size_type num_bonds (const G& g) {
  return boost::num_edges(g);
}

template <class V, class G> 
typename graph_traits<G>::degree_size_type
num_neighbors(const V& v, const G& g) { return boost::out_degree(v,g); }

template <class V, class G> 
std::pair<typename graph_traits<G>::neighbor_bond_iterator,
          typename graph_traits<G>::neighbor_bond_iterator>
neighbor_bonds(const V& v, const G& g) { return boost::out_edges(v,g);}

template <class V, class G> 
std::pair<typename graph_traits<G>::neighbor_iterator,
          typename graph_traits<G>::neighbor_iterator>
neighbors(const V& v, const G& g) { return boost::adjacent_vertices(v,g); }

template <class G>
typename boost::graph_traits<G>::vertex_descriptor
source_impl(const typename boost::graph_traits<G>::edge_descriptor& e,
            const G& g) { return boost::source(e,g); }
  
template <class G>
typename boost::graph_traits<G>::vertex_descriptor
target_impl(const typename boost::graph_traits<G>::edge_descriptor& e,
            const G& g) { return boost::target(e,g); }


template <class V, class G> 
typename graph_traits<G>::site_descriptor 
neighbor (const V& v, typename graph_traits<G>::degree_size_type i, const G& g) 
{ return *(adjacent_vertices(v,g).first+i);}

template <class G>
typename graph_traits<G>::site_descriptor site(typename graph_traits<G>::sites_size_type i, const G& g)
{ return vertex(i,g);}

template <class T>
const typename graph_traits<T>::graph_type& graph(const T& x)
{
  return x.graph();
}

} // end namespace alps

#endif // ALPS_LATTICE_GRAPH_TRAITS_H
