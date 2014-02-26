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
#include <boost/tuple/tuple.hpp>
#include <boost/graph/graph_traits.hpp>

namespace alps {
namespace graph {
namespace detail {
namespace assert_helpers {

template <typename Graph>
bool edge_list_matches_graph(std::vector<typename boost::graph_traits<Graph>::edge_descriptor> edge_list, Graph const& g)
{
    typename boost::graph_traits<Graph>::edge_iterator it,end;
    boost::tie(it,end) = edges(g);
    std::vector<typename boost::graph_traits<Graph>::edge_descriptor> gel(it,end);
    std::sort(edge_list.begin(),edge_list.end());
    std::sort(gel.begin(),gel.end());
    return std::equal(edge_list.begin(),edge_list.end(),gel.begin());
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

} // end namespace assert_helpers
} // end namespace detail
} // end namespace graph
} // end namespace alps


#endif // ALPS_GRAPH_DETAIL_ASSERT_HELPERS_HPP
