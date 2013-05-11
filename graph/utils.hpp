/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013 by Andreas Hehn <hehn@phys.ethz.ch>                          *
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


#ifndef ALPS_GRAPH_UTILS_HPP
#define ALPS_GRAPH_UTILS_HPP

#include <alps/lattice/graphproperties.h>
#include <alps/lattice/propertymap.h>
#include <boost/static_assert.hpp>
#include <cassert>
#include <algorithm>

namespace alps {
namespace graph {
/**
  * Returns a vector listing all edge types which occur in the Graph g.
  */
template <typename Graph>
std::vector<typename has_property<alps::edge_type_t,Graph>::edge_property_type> get_edge_color_list(Graph const& g)
{
    BOOST_STATIC_ASSERT(( has_property<alps::edge_type_t,Graph>::edge_property ));
    std::vector<typename has_property<alps::edge_type_t,Graph>::edge_property_type> edge_colors;
    typename graph_traits<Graph>::edge_iterator e_it, e_end;

    // We expect only a small number of different types
    for(boost::tie(e_it,e_end) = edges(g); e_it != e_end; ++e_it)
    {
        typename has_property<alps::edge_type_t,Graph>::edge_property_type ep = get(alps::edge_type_t(), g, *e_it);
        if( find(edge_colors.begin(),edge_colors.end(),ep) == edge_colors.end() )
            edge_colors.push_back(ep);
    }
    return edge_colors;
}



/**
  * Remaps the edge types of graph g according to the specified map.
  * This function modifies the edge properties of the grap.
  * \param g Graph to modify.
  * \param map A vector that maps an edge type (unsigned int) of the graph to a new edge type (unsigned int).
  */
template <typename Graph>
void remap_edge_types(Graph& g, std::vector<unsigned int> const& map)
{
    BOOST_STATIC_ASSERT((boost::is_same<alps::type_type,unsigned int>::value));
    assert( get_edge_color_list(g).size() == map.size() );
    typename boost::graph_traits<Graph>::edge_iterator it, end;
    for(boost::tie(it,end) = edges(g); it != end; ++it)
    {
        unsigned int type = get(alps::edge_type_t(),g,*it);
        put(alps::edge_type_t(),g,*it,map[type]);
    }
}

} // end namespace graph
} // end namespace alps

#endif //ALPS_GRAPH_UTILS_HPP
