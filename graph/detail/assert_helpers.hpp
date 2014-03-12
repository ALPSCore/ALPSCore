/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2014        by Andreas Hehn <hehn@phys.ethz.ch>                   *
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
#ifndef ALPS_GRAPH_DETAIL_ASSERT_HELPERS_HPP
#define ALPS_GRAPH_DETAIL_ASSERT_HELPERS_HPP

#include <algorithm>
#include <vector>
#include <boost/minmax.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/graph/graph_traits.hpp>

namespace alps {
namespace graph {
namespace detail {
namespace assert_helpers {

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
    struct pair_first_equals_second
    {
        template <typename T1, typename T2>
        bool operator()(std::pair<T1,T2> const& p)
        {
            return p.first == p.second;
        }
    };
    using std::sort;
    using std::adjacent_find;
    using std::find;
    typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
    typename boost::graph_traits<Graph>::edge_iterator it, end;
    std::vector<std::pair<vertex_descriptor,vertex_descriptor> > edge_list;
    edge_list.reserve(num_edges(g));
    for (boost::tie(it, end) = edges(g); it != end; ++it)
        edge_list.push_back(boost::minmax(source(*it,g), target(*it,g)));
    sort(edge_list.begin(),edge_list.end());
    return (find(edge_list.begin(),edge_list.end(), first_equals_second()) == edge_list.end() ) // no loop edge
         && (adjacent_find(edge_list.begin(),edge_list.end()) == edge_list.end()); // no more than one edge between two vertices
}

} // end namespace assert_helpers
} // end namespace detail
} // end namespace graph
} // end namespace alps


#endif // ALPS_GRAPH_DETAIL_ASSERT_HELPERS_HPP
