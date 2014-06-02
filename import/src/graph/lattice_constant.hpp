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

#ifndef ALPS_GRAPH_LATTICE_CONSTANT_HPP
#define ALPS_GRAPH_LATTICE_CONSTANT_HPP

#include <alps/graph/detail/lattice_constant_impl.hpp>

namespace alps {
    namespace graph {

        template<typename Subgraph, typename Graph, typename Lattice> std::size_t lattice_constant(
              Subgraph const & S
            , Graph const & G
            , Lattice const & L
            , typename alps::lattice_traits<Lattice>::cell_descriptor c
        ) {
            typedef typename alps::lattice_traits<Lattice>::size_type cell_index_type;

            // Get the possible translation in the lattice
            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder(dimension(L), std::vector<boost::uint_t<8>::fast>(num_vertices(G), num_vertices(G)));
            detail::build_translation_table(G, L, distance_to_boarder);

            typename partition_type<Subgraph>::type subgraph_orbit = boost::get<2>(canonical_properties(S));

            typename alps::lattice_traits<Lattice>::size_type const cell_id = index(c, L);
            std::size_t unit_cell_size = num_vertices(alps::graph::graph(unit_cell(L)));
            std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V;
            for(unsigned v = 0; v < unit_cell_size; ++v)
                V.push_back(cell_id * unit_cell_size + v);

            boost::false_type no_argument;
            detail::vertex_equal_simple<Subgraph> vertex_equal;
            detail::edge_equal_simple<Subgraph>   edge_equal;
            return detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, unit_cell_size, no_argument, boost::false_type(), vertex_equal, edge_equal, boost::false_type());
        }

        template<typename Subgraph, typename Graph, typename Lattice> std::size_t lattice_constant(
              alps::numeric::matrix<unsigned int> & lw
            , Subgraph const & S
            , Graph const & G
            , Lattice const & L
            , typename alps::lattice_traits<Lattice>::cell_descriptor c
            , typename boost::graph_traits<Subgraph>::vertex_descriptor b
            , typename partition_type<Subgraph>::type const & subgraph_orbit
        ) {
            assert(get<alps::graph::partition>(canonical_properties(S,b)) == subgraph_orbit);
            // Get the possible translation in the lattice
            std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder(dimension(L), std::vector<boost::uint_t<8>::fast>(num_vertices(G), num_vertices(G)));
            detail::build_translation_table(G, L, distance_to_boarder);

            typename alps::lattice_traits<Lattice>::size_type const cell_id = index(c, L);
            std::size_t unit_cell_size = num_vertices(alps::graph::graph(unit_cell(L)));
            std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> V;
            for(unsigned v = 0; v < unit_cell_size; ++v)
                V.push_back(cell_id * unit_cell_size + v);

            detail::vertex_equal_simple<Subgraph> vertex_equal;
            detail::edge_equal_simple<Subgraph>   edge_equal;
            return detail::lattice_constant_impl(S, G, V, distance_to_boarder, subgraph_orbit, unit_cell_size, lw, b, vertex_equal, edge_equal, boost::false_type());
        }
    }
}

#endif //ALPS_GRAPH_LATTICE_CONSTANT_HPP
