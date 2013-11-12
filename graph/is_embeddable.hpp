/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2013 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Andreas Hehn <hehn@phys.ethz.ch>                   *
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

#ifdef USE_LATTICE_CONSTANT_2D
#include <alps/graph/lattice_constant_2d.hpp>
#define ALPS_GRAPH_LATTICE_CONSTANT_HPP
#define ALPS_GRAPH_IS_EMBEDDABLE_HPP
#endif // USE_LATTICE_CONSTANT_2D

#ifndef ALPS_GRAPH_IS_EMBEDDABLE_HPP
#define ALPS_GRAPH_IS_EMBEDDABLE_HPP

#include <alps/graph/detail/lattice_constant_impl.hpp>

namespace alps {
    namespace graph {

        template<typename Subgraph, typename Graph> bool is_embeddable(
              Subgraph const & S
            , Graph const & G
            , typename boost::graph_traits<Graph>::vertex_descriptor v
            , typename partition_type<Subgraph>::type const & subgraph_orbit
        ) {
            assert(get<alps::graph::partition>(canonical_properties(S)) == subgraph_orbit);
            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder;

            try {
                std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V(1, v);
                boost::false_type no_argument;
                detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, 1, no_argument, boost::false_type(), boost::true_type());
                return false;
            } catch (detail::embedding_found e) {
                return true;
            }
        }

        template<typename Subgraph, typename Graph> bool is_embeddable(
              Subgraph const & S
            , Graph const & G
            , typename partition_type<Subgraph>::type const & subgraph_orbit
        ) {
            assert(get<alps::graph::partition>(canonical_properties(S)) == subgraph_orbit);
            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder;

            try {
                typename boost::graph_traits<Graph>::vertex_iterator vt, ve;
                boost::false_type no_argument;
                std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V;
                for (boost::tie(vt, ve) = vertices(G); vt != ve; ++vt) {
                    V.clear();
                    V.push_back(*vt);
                    detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, 1, no_argument, boost::false_type(),boost::true_type());
                }
                return false;
            } catch (detail::embedding_found e) {
                return true;
            }
        }

        /**
          * alps::edge_type_t must be an integer value which can be used as index in a vector.
          */
        template<typename Subgraph, typename Graph> bool is_embeddable(
              Subgraph const & S
            , Graph const & G
            , typename partition_type<Subgraph>::type const & subgraph_orbit
            , typename color_partition<Subgraph>::type const & color_partition
        ) {
            using std::distance;
            assert(get<alps::graph::partition>(canonical_properties(S,color_partition)) == subgraph_orbit);

            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder;

            // Try to embedd directly
            try {
                typename boost::graph_traits<Graph>::vertex_iterator vt, ve;
                boost::false_type no_argument;
                std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V;
                for (boost::tie(vt, ve) = vertices(G); vt != ve; ++vt) {
                    V.clear();
                    V.push_back(*vt);
                    detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, 1, no_argument, boost::false_type(),boost::true_type());
                }
            } catch (detail::embedding_found e) {
                return true;
            }

            // Try to embedd color permutations
            Subgraph Sc(S);
            std::vector<std::size_t> index(color_partition.size());
            for(std::size_t i=0; i < color_partition.size(); ++i)
                index[i] = i;

            typename alps::graph::color_partition<Subgraph>::type::const_iterator const color_partition_begin = color_partition.begin();

            while( std::next_permutation(index.begin(), index.end()))
            {
                if(!detail::mapping_respects_color_partition(index,color_partition))
                    continue;
                // map colors
                typename boost::graph_traits<Subgraph>::edge_iterator Sc_it,Sc_end;
                typename boost::graph_traits<Subgraph>::edge_iterator S_it,S_end;
                boost::tie(Sc_it,Sc_end) = edges(Sc);
                boost::tie(S_it,S_end)   = edges(S);
                for(; Sc_it != Sc_end; ++Sc_it, ++S_it)
                {
                    typename alps::has_property<alps::edge_type_t,Subgraph>::edge_property_type const newcolor
                        = (color_partition_begin + index[ distance(color_partition_begin,color_partition.find(get(alps::edge_type_t(),S)[*S_it])) ])->first;
                    get(alps::edge_type_t(),Sc)[*Sc_it] = newcolor;
                }

                assert(get<alps::graph::partition>(canonical_properties(Sc,color_partition)) == subgraph_orbit);
                // Try to embedd
                try {
                    typename boost::graph_traits<Graph>::vertex_iterator vt, ve;
                    boost::false_type no_argument;
                    std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V;
                    for (boost::tie(vt, ve) = vertices(G); vt != ve; ++vt) {
                        V.clear();
                        V.push_back(*vt);
                        detail::lattice_constant_impl(Sc, G, V, distance_to_boarder, subgraph_orbit, 1, no_argument, boost::false_type(),boost::true_type());
                    }
                } catch (detail::embedding_found e) {
                    return true;
                }
            }
            return false;
        }

    }
}

#endif // ALPS_GRAPH_IS_EMBEDDABLE_HPP
