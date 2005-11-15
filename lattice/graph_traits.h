/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_LATTICE_GRAPH_TRAITS_H
#define ALPS_LATTICE_GRAPH_TRAITS_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

namespace alps {

template<class G>
struct graph_traits : public boost::graph_traits<G>
{
  typedef G graph_type;
  typedef typename boost::graph_traits<graph_type>::vertex_iterator
    site_iterator;
  typedef typename boost::graph_traits<graph_type>::edge_iterator
    bond_iterator;
  typedef typename boost::graph_traits<graph_type>::out_edge_iterator
    neighbor_bond_iterator;
  typedef typename boost::graph_traits<graph_type>::vertex_descriptor
    site_descriptor;
  typedef typename boost::graph_traits<graph_type>::edge_descriptor
    bond_descriptor;
  typedef typename boost::graph_traits<graph_type>::vertices_size_type
    sites_size_type;
  typedef typename boost::graph_traits<graph_type>::edges_size_type
    bonds_size_type;
  typedef typename boost::graph_traits<graph_type>::degree_size_type
    neighbors_size_type;
  typedef typename boost::graph_traits<graph_type>::adjacency_iterator
    neighbor_iterator;
};

} // end namespace alps

namespace boost {

// site-bond wrapper for boost::adjacency_list

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
  sites_size_type
num_sites(const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{ return num_vertices(g); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
  bonds_size_type
num_bonds(const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{ return num_edges(g); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
std::pair<
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    site_iterator,
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    site_iterator>
sites(const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{ return vertices(g); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
  site_descriptor
site(
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    sites_size_type i,
  const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g) 
{ return vertex(i,g); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
std::pair<
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    bond_iterator,
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    bond_iterator>
bonds(const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{ return edges(g); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
  bond_descriptor
bond(
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    bonds_size_type i,
  const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g) 
{ return *(bonds(g).first+i); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
  degree_size_type
num_neighbors(
  const typename alps::graph_traits<
    adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::site_descriptor& v,
  const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{ return out_degree(v, g); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
std::pair<
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    neighbor_bond_iterator,
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    neighbor_bond_iterator>
neighbor_bonds(
  const typename alps::graph_traits<
    adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::site_descriptor& v,
  const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{ return out_edges(v,g); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
std::pair<
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    neighbor_iterator,
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    neighbor_iterator>
neighbors(
  const typename alps::graph_traits<
    adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::site_descriptor& v,
  const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g)
{ return adjacent_vertices(v,g); }

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
  site_descriptor
neighbor(
  const typename alps::graph_traits<
    adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::site_descriptor& v,
  typename alps::graph_traits<adjacency_list<T0, T1, T2, T3, T4, T5, T6> >::
    degree_size_type i,
  const adjacency_list<T0, T1, T2, T3, T4, T5, T6>& g) 
{ return *(adjacent_vertices(v, g).first+i); }

} // end namespace boost

namespace alps {

namespace detail {

// wapper for vertex-edge functions

template<class G>
typename graph_traits<G>::vertices_size_type
num_vertices_wrap(const G& g) { return num_vertices(g); }

template<class G>
typename graph_traits<G>::edges_size_type
num_edges_wrap(const G& g) { return num_edges(g); }

template<class G>
std::pair<typename graph_traits<G>::vertex_iterator,
          typename graph_traits<G>::vertex_iterator>
vertices_wrap(const G& g) { return vertices(g); }

template<class G>
std::pair<typename graph_traits<G>::edge_iterator,
          typename graph_traits<G>::edge_iterator>
edges_wrap(const G& g) { return edges(g); }

template<class G>
typename graph_traits<G>::degree_size_type
out_degree_wrap(const typename graph_traits<G>::vertex_descriptor& v,
                const G& g) { return out_degree(v, g); }

template<class G>
typename graph_traits<G>::degree_size_type
in_degree_wrap(const typename graph_traits<G>::vertex_descriptor& v,
               const G& g) { return in_degree(v, g); }

template<class G>
typename graph_traits<G>::degree_size_type
degree_wrap(const typename graph_traits<G>::vertex_descriptor& v,
            const G& g) { return degree(v, g); }

template<class G>
std::pair<typename graph_traits<G>::out_edge_iterator,
          typename graph_traits<G>::out_edge_iterator>
out_edges_wrap(const typename graph_traits<G>::vertex_descriptor& v,
               const G& g) { return out_edges(v, g); }

template<class G>
std::pair<typename graph_traits<G>::in_edge_iterator,
          typename graph_traits<G>::in_edge_iterator>
in_edges_wrap(const typename graph_traits<G>::vertex_descriptor& v,
              const G& g) { return in_edges(v, g); }

template<class G>
std::pair<typename graph_traits<G>::adjacency_iterator,
          typename graph_traits<G>::adjacency_iterator>
adjacent_vertices_wrap(
  const typename graph_traits<G>::vertex_descriptor& v,
  const G& g) { return adjacent_vertices(v, g); }

template<class G>
typename graph_traits<G>::vertex_descriptor
vertex_wrap(typename graph_traits<G>::vertex_size_type i,
          const G& g) { return vertex(i, g); }

template <class G>
typename graph_traits<G>::vertex_descriptor
source_wrap(const typename graph_traits<G>::edge_descriptor& e,
            const G& g) { return source(e, g); }
  
template <class G>
typename graph_traits<G>::vertex_descriptor
target_wrap(const typename graph_traits<G>::edge_descriptor& e,
            const G& g) { return target(e, g); }

// wrapper for site-bond functions

template<class G>
typename graph_traits<G>::sites_size_type
num_sites_wrap(const G& g) { return num_sites(g); }

template<class G>
typename graph_traits<G>::bonds_size_type
num_bonds_wrap(const G& g) { return num_bonds(g); }

template<class G>
std::pair<typename graph_traits<G>::site_iterator,
          typename graph_traits<G>::site_iterator>
sites_wrap(const G& g) { return sites(g); }

template<class G>
typename graph_traits<G>::site_descriptor
site_wrap(typename graph_traits<G>::sites_size_type i,
          const G& g) { return site(i, g); }

template<class G>
std::pair<typename graph_traits<G>::bond_iterator,
          typename graph_traits<G>::bond_iterator>
bonds_wrap(const G& g) { return bonds(g); }

template<class G>
typename graph_traits<G>::bond_descriptor
bond_wrap(typename graph_traits<G>::bonds_size_type i,
          const G& g) { return bond(i, g); }

template<class G>
typename graph_traits<G>::degree_size_type
num_neighbors_wrap(const typename graph_traits<G>::site_descriptor& v,
                   const G& g) { return num_neighbors(v, g); }

template<class G>
std::pair<typename graph_traits<G>::neighbor_bond_iterator,
          typename graph_traits<G>::neighbor_bond_iterator>
neighbor_bonds_wrap(const typename graph_traits<G>::site_descriptor& v,
                    const G& g) { return neighbor_bonds(v, g); }

template<class G>
std::pair<typename graph_traits<G>::neighbor_iterator,
          typename graph_traits<G>::neighbor_iterator>
neighbors_wrap(const typename graph_traits<G>::site_descriptor& v,
               const G& g) { return neighbors(v, g); }

template<class G>
typename graph_traits<G>::site_descriptor
neighbor_wrap(const typename graph_traits<G>::site_descriptor& v,
              typename graph_traits<G>::degree_size_type i,
              const G& g) { return neighbor(v, i, g); }

template <class T>
typename T::graph_type& graph_wrap(T& x)
{ return x.graph(); }

template <class T>
const typename T::graph_type& graph_wrap(const T& x)
{ return x.graph(); }

} // end namespace detail

} // end namespace alps

#endif // ALPS_LATTICE_GRAPH_TRAITS_H
