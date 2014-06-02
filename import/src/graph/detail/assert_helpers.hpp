/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_GRAPH_DETAIL_ASSERT_HELPERS_HPP
#define ALPS_GRAPH_DETAIL_ASSERT_HELPERS_HPP

#include <algorithm>
#include <vector>
#include <boost/algorithm/minmax.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/graph/graph_traits.hpp>

namespace alps {
namespace graph {
namespace detail {
namespace assert_helpers {

struct tuple_0th_equals_1st
{
    template <typename T1, typename T2>
    bool operator()(boost::tuple<T1,T2> const& t) { return get<0>(t) == get<1>(t); }
};

template <typename Graph>
bool edge_list_matches_graph(std::vector<typename boost::graph_traits<Graph>::edge_descriptor> edge_list, Graph const& g)
{
    using std::sort;
    using std::equal;
    typename boost::graph_traits<Graph>::edge_iterator it,end;
    boost::tie(it,end) = edges(g);
    std::vector<typename boost::graph_traits<Graph>::edge_descriptor> gel(it,end);
    sort(edge_list.begin(),edge_list.end());
    sort(gel.begin(),gel.end());
    return equal(edge_list.begin(),edge_list.end(),gel.begin());
}

template <typename Graph>
bool color_partitions_are_complete(typename color_partition<Graph>::type const& color_partition, Graph const& g)
{
    // Check if all edge colors occuring in the graph are also in the color_partition
    typename boost::graph_traits<Graph>::edge_iterator it, end;
    bool are_complete = true;
    for (boost::tie(it, end) = edges(g); it != end; ++it)
        are_complete = are_complete && color_partition.find(get(alps::edge_type_t(),g)[*it]) != color_partition.end();
    return are_complete;
}

//
// Checks if graph g is a simple graph.
// A simple graph is a graph where
//   - no edge is connected to the same vertex at both ends and
//   - no two vertices are directly connected by more than one edge.
//
template <typename Graph>
bool is_simple_graph(Graph const& g)
{
    using std::sort;
    using std::adjacent_find;
    using std::find_if;
    typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
    typename boost::graph_traits<Graph>::edge_iterator it, end;
    std::vector<boost::tuple<vertex_descriptor,vertex_descriptor> > edge_list;
    edge_list.reserve(num_edges(g));
    for (boost::tie(it, end) = edges(g); it != end; ++it)
        edge_list.push_back(boost::minmax(source(*it,g), target(*it,g)));
    sort(edge_list.begin(),edge_list.end());
    return (find_if(edge_list.begin(),edge_list.end(), tuple_0th_equals_1st()) == edge_list.end() ) // no loop edge
         && (adjacent_find(edge_list.begin(),edge_list.end()) == edge_list.end()); // no more than one edge between two vertices
}

} // end namespace assert_helpers
} // end namespace detail
} // end namespace graph
} // end namespace alps


#endif // ALPS_GRAPH_DETAIL_ASSERT_HELPERS_HPP
