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
namespace detail {
    template <typename Graph, typename PropertyTag>
    struct iteration_selector;

    template <typename Graph>
    struct iteration_selector<Graph,alps::vertex_type_t>
    {
        typedef typename boost::graph_traits<Graph>::vertex_iterator iterator;
        static std::pair<iterator,iterator> range(Graph const& g) { return vertices(g); }
    };

    template <typename Graph>
    struct iteration_selector<Graph,alps::edge_type_t>
    {
        typedef typename boost::graph_traits<Graph>::edge_iterator iterator;
        static std::pair<iterator,iterator> range(Graph const& g) { return edges(g); }
    };
}

/**
  * Extracts a sorted list of all vertex or edge colors in a graph.
  * \parm tag Selects if vertex or edge colors should be listed, it is either alps::vertex_type_t or alps::edge_type_t.
  * \parm g The graph to be inspected.
  * \return A vector containing allcolors that occur in the graph in ascending order.
  */
template <typename Graph, typename PropertyTag>
std::vector<typename boost::property_map<Graph,PropertyTag>::type::value_type> get_color_list(PropertyTag tag, Graph const& g)
{
    BOOST_STATIC_ASSERT(( boost::is_same<PropertyTag,alps::edge_type_t>::value || boost::is_same<PropertyTag,alps::vertex_type_t>::value ));
    BOOST_STATIC_ASSERT(( has_property<PropertyTag,Graph>::edge_property || has_property<PropertyTag,Graph>::vertex_property ));
    using std::sort;
    std::vector<typename boost::property_map<Graph,PropertyTag>::type::value_type> colors;
    typename boost::graph_traits<Graph>::edge_iterator e_it, e_end;

    typename detail::iteration_selector<Graph,PropertyTag>::iterator it,end;

    // We expect only a small number of different types
    for(boost::tie(it,end) = detail::iteration_selector<Graph,PropertyTag>::range(g); it != end; ++it)
    {
        typename boost::property_map<Graph,PropertyTag>::type::value_type ep = get(tag, g, *it);
        if( find(colors.begin(),colors.end(),ep) == colors.end() )
            colors.push_back(ep);
    }
    sort(colors.begin(),colors.end());
    return colors;
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
    assert( get_color_list(alps::edge_type_t(),g).size() == map.size() );
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
