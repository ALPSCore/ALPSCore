/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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

            detail::vertex_equal_simple<Subgraph>   vertex_equal;
            detail::edge_equal_simple<Subgraph>     edge_equal;
            try {
                std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V(1, v);
                boost::false_type no_argument;
                detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, 1, no_argument, boost::false_type(), vertex_equal, edge_equal, boost::true_type());
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

            detail::vertex_equal_simple<Subgraph>   vertex_equal;
            detail::edge_equal_simple<Subgraph>     edge_equal;
            try {
                typename boost::graph_traits<Graph>::vertex_iterator vt, ve;
                boost::false_type no_argument;
                std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V;
                for (boost::tie(vt, ve) = vertices(G); vt != ve; ++vt) {
                    V.clear();
                    V.push_back(*vt);
                    detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, 1, no_argument, boost::false_type(), vertex_equal, edge_equal, boost::true_type());
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
            assert(get<alps::graph::partition>(canonical_properties(S)) == subgraph_orbit);
            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder;

            detail::vertex_equal_simple<Subgraph>               vertex_equal;
            detail::edge_equal_with_color_symmetries<Subgraph>  edge_equal(color_partition);
            try {
                typename boost::graph_traits<Graph>::vertex_iterator vt, ve;
                boost::false_type no_argument;
                std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V;
                for (boost::tie(vt, ve) = vertices(G); vt != ve; ++vt) {
                    edge_equal.reset();
                    V.clear();
                    V.push_back(*vt);
                    detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, 1, no_argument, boost::false_type(), vertex_equal, edge_equal, boost::true_type());
                }
                return false;
            } catch (detail::embedding_found e) {
                return true;
            }
        }

    }
}

#endif // ALPS_GRAPH_IS_EMBEDDABLE_HPP
