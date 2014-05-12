/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Andreas Hehn <hehn@phys.ethz.ch>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_GRAPH_CANONICAL_GRAPH_HPP
#define ALPS_GRAPH_CANONICAL_GRAPH_HPP
#include <alps/graph/canonical_properties.hpp>
#include <boost/graph/graph_traits.hpp>
#include <cassert>

namespace alps {
namespace graph {

template <typename Graph>
class canonical_graph
{
    public:

        typedef Graph graph_type;
        explicit canonical_graph(Graph const& g)
            : graph_(g), properties_(canonical_properties(g))
        {
        }
        
        canonical_graph(Graph const& g, typename canonical_properties_type<Graph>::type const& p)
            : graph_(g), properties_(p)
        {
            assert(p == alps::graph::canonical_properties(g));
        }

        operator Graph const() const
        {
            return graph_;
        }

        graph_type const& graph() const
        {
            return graph_;
        }

        bool operator == (canonical_graph const& g) const
        {
            return get<label>(properties_) == get<label>(g.properties_);
        }

        bool operator < (canonical_graph const& g) const
        {
            return get<label>(properties_) < get<label>(g.properties_);
        }

        typename graph_label<Graph>::type const& get_properties() const
        {
            return properties_;
        }

        typename graph_label<Graph>::type get_label() const
        {
            return get<label>(properties_);
        }
        
        typename partition_type<Graph>::type const& get_partition() const
        {
            return get<partition>(properties_);
        }

    private:
        // TODO check const
        Graph graph_;
        typename canonical_properties_type<Graph>::type properties_;
};

template <typename Graph>
std::ostream& operator << (std::ostream& os, alps::graph::canonical_graph<Graph> const& g)
{
    os<<g.graph();
    return os;
} 

template <typename Graph>
typename graph_traits<canonical_graph<Graph> >::edges_size_type
num_edges(canonical_graph<Graph> const& g){
    return num_edges(g.graph());
}

template <typename Graph>
std::pair<typename graph_traits<canonical_graph<Graph> >::edge_descriptor,bool>
edge(
      typename graph_traits<canonical_graph<Graph> >::vertex_descriptor u
    , typename graph_traits<canonical_graph<Graph> >::vertex_descriptor v
    , canonical_graph<Graph> const& g
    ){
    return edge(u,v,g.graph());
}

template <typename Graph>
std::pair<typename graph_traits<canonical_graph<Graph> >::edge_iterator, typename graph_traits<canonical_graph<Graph> >::edge_iterator>
edges( canonical_graph<Graph> const& g) {
    return edges(g.graph());
}

template <typename Graph>
typename graph_traits<canonical_graph<Graph> >::vertex_descriptor
source(typename graph_traits<canonical_graph<Graph> >::edge_descriptor e, canonical_graph<Graph> const& g) {
    return source(e,g.graph());
}

template <typename Graph>
typename graph_traits<canonical_graph<Graph> >::vertex_descriptor
target(typename graph_traits<canonical_graph<Graph> >::edge_descriptor e, canonical_graph<Graph> const& g) {
    return target(e,g.graph());
}

template <typename Graph>
typename graph_traits<canonical_graph<Graph> >::vertices_size_type
num_vertices(canonical_graph<Graph> const& g){
    return num_vertices(g.graph());
}
template <typename Graph>
std::pair<typename graph_traits<canonical_graph<Graph> >::vertex_iterator, typename graph_traits<canonical_graph<Graph> >::vertex_iterator>
vertices(canonical_graph<Graph> const& g) {
    return vertices(g.graph());
}

template <typename Graph>
typename graph_traits<canonical_graph<Graph> >::degree_size_type
degree(
      typename graph_traits<canonical_graph<Graph> >::vertex_descriptor v
    , canonical_graph<Graph> const& g
    ){
    return degree(v,g.graph());
}

template <typename Graph>
typename graph_traits<canonical_graph<Graph> >::degree_size_type
out_degree(
      typename graph_traits<canonical_graph<Graph> >::vertex_descriptor v
    , canonical_graph<Graph> const& g
    ){
    return out_degree(v,g.graph());
}

template <typename Graph>
std::pair<typename graph_traits<canonical_graph<Graph> >::adjacency_iterator, typename graph_traits<canonical_graph<Graph> >::adjacency_iterator>
adjacent_vertices(
          typename graph_traits<canonical_graph<Graph> >::vertex_descriptor v
        , canonical_graph<Graph> const& g
        ){
    return adjacent_vertices(v,g.graph());
}

using boost::property_map;
template <typename Graph, typename Tag>
typename property_map<Graph, Tag>::const_type
get(Tag p, canonical_graph<Graph> const& g) {
    return get(p,g.graph());
}

} // namespace graph
} // namspace alps


// boost::graph specializations for alps::graph::canonical_graph
namespace boost {

template <typename Graph>
struct graph_traits<alps::graph::canonical_graph<Graph> > {
    typedef typename graph_traits<Graph>::vertex_descriptor      vertex_descriptor;
    typedef typename graph_traits<Graph>::edge_descriptor        edge_descriptor;
    typedef typename graph_traits<Graph>::adjacency_iterator     adjacency_iterator;
    typedef typename graph_traits<Graph>::out_edge_iterator      out_edge_iterator;
    typedef typename graph_traits<Graph>::in_edge_iterator       in_edge_iterator;
    typedef typename graph_traits<Graph>::vertex_iterator        vertex_iterator;
    typedef typename graph_traits<Graph>::edge_iterator          edge_iterator;
    typedef typename graph_traits<Graph>::directed_category      directed_category;
    typedef typename graph_traits<Graph>::edge_parallel_category edge_parallel_category;
    typedef typename graph_traits<Graph>::traversal_category     traversal_category;
    typedef typename graph_traits<Graph>::vertices_size_type     vertices_size_type;
    typedef typename graph_traits<Graph>::edges_size_type        edges_size_type;
    typedef typename graph_traits<Graph>::degree_size_type       degree_size_type;
};

template <typename Graph>
struct graph_property_type<alps::graph::canonical_graph<Graph> >{
    typedef typename graph_property_type<Graph>::type type;
};
template <typename Graph>
struct edge_property_type<alps::graph::canonical_graph<Graph> >{
    typedef typename edge_property_type<Graph>::type type;
};
template <typename Graph>
struct vertex_property_type<alps::graph::canonical_graph<Graph> >{
    typedef typename vertex_property_type<Graph>::type type;
};

} // end namespace boost

#endif //ALPS_GRAPH_CANONICAL_GRAPH_HPP
