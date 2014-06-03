/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_GRAPH_DETAIL_LATTICE_CONSTANT_IMPL_HPP
#define ALPS_GRAPH_DETAIL_LATTICE_CONSTANT_IMPL_HPP

#include <alps/ngs/stacktrace.hpp>

#include <alps/lattice.h>
#include <alps/numeric/vector_functions.hpp>
#include <alps/graph/canonical_properties.hpp>
#include <alps/numeric/matrix.hpp>

#include <boost/array.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>
#include <boost/integer_traits.hpp>

#include <deque>
#include <vector>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#if !defined(USE_COMPRESSED_EMBEDDING) && !defined(USE_COMPRESSED_EMBEDDING2) && !defined(USE_GENERIC_EMBEDDING)
    #define USE_GENERIC_EMBEDDING
#endif

namespace alps {
    namespace graph {

        namespace detail {

            struct embedding_generic_type {

                embedding_generic_type(std::size_t vertices_size, std::size_t edges_size)
                    : hash(0)
                    , counter(new boost::uint8_t(1))
                    , vertices(new std::vector<std::vector<boost::uint16_t> >(vertices_size))
                    , edges(new std::vector<boost::uint64_t>((edges_size >> 6) + ((edges_size & 0x3F) == 0 ? 0 : 1)))
                {}

                embedding_generic_type(embedding_generic_type const & rhs)
                    : hash(rhs.hash)
                    , counter(rhs.counter)
                    , vertices(rhs.vertices)
                    , edges(rhs.edges)
                {
                    assert(*counter < boost::integer_traits<boost::uint8_t>::const_max - 1);
                    ++*counter;
                }

                ~embedding_generic_type() {
                    if (!--*counter) {
                        delete counter;
                        delete vertices;
                        delete edges;
                    }
                }

                bool operator == (embedding_generic_type const & rhs) const {
                    return hash == rhs.hash
                        && *edges == *rhs.edges
                        && *vertices == *rhs.vertices
                    ;
                }

                std::size_t hash;
                boost::uint8_t * counter;
                std::vector<std::vector<boost::uint16_t> > * vertices;
                std::vector<boost::uint64_t> * edges;

                private:
                    embedding_generic_type() {}
            };

            std::size_t hash_value(embedding_generic_type const & value) {
                return value.hash;
            }

            template <typename Graph, typename Lattice> void build_translation_table(
                  Graph const & graph
                , Lattice const & lattice
                , std::vector<std::vector<boost::uint_t<8>::fast> > & distance_to_boarder
            ) {
                typedef typename alps::lattice_traits<Lattice>::cell_iterator cell_iterator;
                typedef typename alps::lattice_traits<Lattice>::offset_type offset_type;
                typedef typename alps::lattice_traits<Lattice>::size_type cell_index_type;

                std::vector<std::vector<unsigned> > translations(dimension(lattice), std::vector<unsigned>(num_vertices(graph), num_vertices(graph)));
                unsigned vtcs_per_ucell = num_vertices(alps::graph::graph(unit_cell(lattice)));
                for(std::size_t d = 0; d < dimension(lattice); ++d) {
                    for(std::pair<cell_iterator,cell_iterator> c = cells(lattice); c.first != c.second; ++c.first) {
                        offset_type ofst = offset(*c.first,lattice);
                        offset_type move(dimension(lattice));
                        move[d] = -1;
                        std::pair<bool,bool> on_lattice_pbc_crossing = shift(ofst,move,lattice);
                        if(on_lattice_pbc_crossing.first && !on_lattice_pbc_crossing.second) {
                            const cell_index_type cellidx = index(*c.first,lattice);
                            const cell_index_type neighboridx = index(cell(ofst, lattice), lattice);
                            for(unsigned v = 0; v < vtcs_per_ucell; ++v)
                                translations[d][cellidx * vtcs_per_ucell + v] = neighboridx * vtcs_per_ucell + v;
                        }
                    }
                    unsigned v;
                    for (std::vector<unsigned>::const_iterator it = translations[d].begin(); it != translations[d].end(); ++it) {
                        if (*it != num_vertices(graph))
                        {
                            distance_to_boarder[d][v = *it] = 0;
                            while ((v = translations[d][v]) != num_vertices(graph))
                                ++distance_to_boarder[d][*it];
                        }
                    }
                }
            }

            template<typename GeometricInfo, typename Graph, typename Subgraph, typename BreakingVertex> typename boost::disable_if<
                boost::is_same<GeometricInfo, boost::false_type>
            >::type lattice_constant_geometry(
                  GeometricInfo & geometric_info
                , Subgraph const & S
                , Graph const & G
                , std::vector<std::size_t> const & I
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , BreakingVertex const & breaking_vertex
                , bool inserted
            ) {
                if (inserted)
                    for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it) {
                        bool is_valid = true;
                        for(std::size_t d = 0; is_valid && d < distance_to_boarder.size(); ++d)
                            is_valid = distance_to_boarder[d][pinning[breaking_vertex]] <= distance_to_boarder[d][*it];
                        if (is_valid)
                            geometric_info(*it - pinning[breaking_vertex],I[it - pinning.begin()])++;
                    }
            }

            template<typename GeometricInfo, typename Graph, typename Subgraph, typename BreakingVertex> typename boost::enable_if<
                boost::is_same<GeometricInfo, boost::false_type>
            >::type lattice_constant_geometry(
                  GeometricInfo &
                , Subgraph const & S
                , Graph const &
                , std::vector<std::size_t> const &
                , std::vector<std::vector<boost::uint_t<8>::fast> > const &
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const &
                , typename partition_type<Subgraph>::type const &
                , BreakingVertex const &
                , bool
            ) {}

            struct embedding_found {};

            // TODO: move back into main function after optimizing
            template<typename Subgraph, typename Graph, unsigned SubVertexNum, unsigned CoordNum, typename GeometricInfo, typename BreakingVertex> void lattice_constant_insert(
                  Subgraph const & S
                , Graph const & G
                , std::vector<std::size_t> const & I
                , boost::unordered_set<embedding_generic_type> & matches
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , std::size_t unit_cell_size
                , GeometricInfo & geometric_info
                , BreakingVertex
                , boost::true_type
            ) {
                throw embedding_found();
            }

            // TODO: move back into main function after optimizing
            template<typename Subgraph, typename Graph, unsigned SubVertexNum, unsigned CoordNum, typename GeometricInfo, typename BreakingVertex> void lattice_constant_insert(
                  Subgraph const & S
                , Graph const & G
                , std::vector<std::size_t> const & I
                , boost::unordered_set<embedding_generic_type> & matches
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , std::size_t unit_cell_size
                , GeometricInfo & geometric_info
                , BreakingVertex const & breaking_vertex
                // TODO: make argument, to pass SubVertexNum and CoordNum, so no explicit call is needed ...
                , boost::false_type
            ) {
                embedding_generic_type embedding_generic(subgraph_orbit.size(), num_vertices(S) * (num_vertices(S) + 1) / 2);

                for (std::vector<std::vector<boost::uint16_t> >::iterator it = embedding_generic.vertices->begin(); it != embedding_generic.vertices->end(); ++it)
                    it->reserve(subgraph_orbit[it - embedding_generic.vertices->begin()].size());

                std::size_t bits_per_dim = 0;
                while ((0x01u << ++bits_per_dim) < num_vertices(S));
                assert((0x01u << (distance_to_boarder.size() * bits_per_dim)) < boost::integer_traits<boost::uint16_t>::const_max);

                std::vector<boost::uint_t<8>::fast> distances(distance_to_boarder.size(), num_vertices(G));
                for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it)
                    for(std::size_t d = 0; d < distance_to_boarder.size(); ++d)
                        distances[d] = std::min(distances[d], distance_to_boarder[d][*it]);

                std::vector<boost::uint16_t> lattice_pinning(pinning.size());
                for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it) {
                    lattice_pinning[it - pinning.begin()] = *it % unit_cell_size;
                    for(std::size_t d = 0; d < distance_to_boarder.size(); ++d) {
                        lattice_pinning[it - pinning.begin()] <<= bits_per_dim;
                        lattice_pinning[it - pinning.begin()] += distance_to_boarder[d][*it] - distances[d];
                    }
                    (*embedding_generic.vertices)[I[it - pinning.begin()]].push_back(lattice_pinning[it - pinning.begin()]);
                }
                for (std::vector<std::vector<boost::uint16_t> >::iterator it = embedding_generic.vertices->begin(); it != embedding_generic.vertices->end(); ++it) {
                    using boost::hash_combine;
                    std::sort(it->begin(), it->end());
                    for (std::vector<boost::uint16_t>::const_iterator jt = it->begin(); jt != it->end(); ++jt)
                        hash_combine(embedding_generic.hash, *jt);
                }

                for (std::vector<boost::uint16_t>::iterator it = lattice_pinning.begin(); it != lattice_pinning.end(); ++it) {
                    std::vector<boost::uint16_t>::iterator jt = (*embedding_generic.vertices)[I[it - lattice_pinning.begin()]].begin();
                    for (; *jt != *it; ++jt);
                    *it = jt - (*embedding_generic.vertices)[I[it - lattice_pinning.begin()]].begin();
                    for (std::size_t i = 0; i < I[it - lattice_pinning.begin()]; ++i)
                        *it += (*embedding_generic.vertices)[i].size();
                }

                typename boost::graph_traits<Subgraph>::edge_iterator s_ei, s_ee;
                for (boost::tie(s_ei, s_ee) = edges(S); s_ei != s_ee; ++s_ei) {
                    std::size_t v1 = std::min(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
                    std::size_t v2 = std::max(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
                    std::size_t index = v1 * num_vertices(S) - (v1 - 1) * v1 / 2 + v2 - v1;
                    (*embedding_generic.edges)[index >> 6] |= 0x01 << (index & 0x3F);
                }
                for (std::vector<boost::uint64_t>::const_iterator it = embedding_generic.edges->begin(); it != embedding_generic.edges->end(); ++it) {
                    using boost::hash_combine;
                    hash_combine(embedding_generic.hash, *it);
                }

                lattice_constant_geometry(geometric_info, S, G, I, distance_to_boarder, pinning, subgraph_orbit, breaking_vertex, matches.insert(embedding_generic).second);
            }

            template <typename Subgraph>
            struct edge_equal_simple
            {
              private:
                template <typename Graph>
                bool impl(
                  typename boost::graph_traits<Subgraph>::edge_descriptor const & s_e
                , typename boost::graph_traits<Graph>::edge_descriptor const & g_e
                , Subgraph const & S
                , Graph const & G
                , boost::mpl::true_
                ) const {
                    return get(alps::edge_type_t(), S)[s_e] == get(alps::edge_type_t(), G)[g_e];
                }

                template <typename Graph>
                bool impl(
                  typename boost::graph_traits<Subgraph>::edge_descriptor const & s_e
                , typename boost::graph_traits<Graph>::edge_descriptor const & g_e
                , Subgraph const & S
                , Graph const & G
                , boost::mpl::false_
                ) const {
                    return true;
                }

              public:
                template <typename Graph>
                bool operator()(
                  typename boost::graph_traits<Subgraph>::edge_descriptor const & s_e
                , typename boost::graph_traits<Graph>::edge_descriptor const & g_e
                , Subgraph const & S
                , Graph const & G
                ) const {
                    return impl(s_e,g_e,S,G,boost::mpl::bool_<has_property<alps::edge_type_t,Subgraph>::edge_property>());
                }

            };

            template <typename Subgraph>
            struct edge_equal_with_color_symmetries
            {
                typedef typename boost::property_map<Subgraph,alps::edge_type_t>::type::value_type color_type;
                typedef typename color_partition<Subgraph>::type                                color_partition_type;
                typedef std::vector<color_type>                                                 color_map_type;

                static color_type const invalid = boost::integer_traits<color_type>::const_max;

                edge_equal_with_color_symmetries(color_partition_type const& color_partition)
                : color_partition_(color_partition), color_map_(color_partition_.size(),color_type(invalid))
                {
                    BOOST_STATIC_ASSERT(( has_property<alps::edge_type_t,Subgraph>::edge_property ));
                }

                void reset()
                {
                    color_map_.clear();
                    color_map_.resize(color_partition_.size(),invalid);
                }


                template <typename Graph>
                bool operator()(
                      typename boost::graph_traits<Subgraph>::edge_descriptor const & s_e
                    , typename boost::graph_traits<Graph>::edge_descriptor const & g_e
                    , Subgraph const & S
                    , Graph const & G
                ) {
                    color_type const sec = get(alps::edge_type_t(), S)[s_e];
                    color_type const gec = get(alps::edge_type_t(), G)[g_e];
                    assert(sec < color_map_.size());

                    if(color_map_[sec] == invalid && color_partition_[sec] == color_partition_[gec])
                    {
                        // Try to add a mapping from color sec to color gec to the color_map_ while keeping the mapping unique.
                        // (i.e. no two colors sec0,sec1 can be mapped to the same color gec)
                        if(std::find(color_map_.begin(),color_map_.end(), gec) == color_map_.end())
                            color_map_[sec] = gec;
                    }
                    return color_map_[sec] == gec;
                }
            private:
                color_partition_type    color_partition_;
                color_map_type          color_map_;
            };

            template <typename Subgraph>
            struct vertex_equal_simple
            {
              private:
                template <typename Graph>
                bool impl(
                  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s_v
                , typename boost::graph_traits<Graph>::vertex_descriptor const & g_v
                , Subgraph const & S
                , Graph const & G
                , boost::mpl::true_
                ) const {
                    return get(alps::vertex_type_t(), S)[s_v] == get(alps::vertex_type_t(), G)[g_v];
                }

                template <typename Graph>
                bool impl(
                  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s_v
                , typename boost::graph_traits<Graph>::vertex_descriptor const & g_v
                , Subgraph const & S
                , Graph const & G
                , boost::mpl::false_
                ) const {
                    return true;
                }

              public:
                template <typename Graph>
                bool operator()(
                  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s_v
                , typename boost::graph_traits<Graph>::vertex_descriptor const & g_v
                , Subgraph const & S
                , Graph const & G
                ) const {
                    return impl(s_v,g_v,S,G,boost::mpl::bool_<has_property<alps::vertex_type_t,Subgraph>::vertex_property>());
                }
            };

            // TODO: make an object out of walker
            template<typename Subgraph, typename Graph, typename GeometricInfo, typename BreakingVertex, typename VertexEqual, typename EdgeEqual, typename ExitOnMatch> void lattice_constant_walker(
                  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s
                , typename boost::graph_traits<Graph>::vertex_descriptor const & g
                , Subgraph const & S
                , Graph const & G
                , std::vector<std::size_t> const & I
                , boost::unordered_set<embedding_generic_type> & matches
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , std::deque<std::pair<
                      typename boost::graph_traits<Subgraph>::vertex_descriptor
                    , typename boost::graph_traits<Graph>::vertex_descriptor
                  > > stack
                , boost::dynamic_bitset<> placed
                , boost::dynamic_bitset<> & visited
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> & pinning
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , std::size_t unit_cell_size
                , GeometricInfo & geometric_info
                , BreakingVertex const & breaking_vertex
                , VertexEqual & vertex_equal
                , EdgeEqual & edge_equal
                , ExitOnMatch exit_on_match
            ) {
                typedef typename boost::graph_traits<Subgraph>::vertex_descriptor SubgraphVertex;
                typedef typename boost::graph_traits<Graph>::vertex_descriptor GraphVertex;

                if (out_degree(s, S) > out_degree(g, G))
                    return;
                if (!vertex_equal(s, g, S, G))
                    return;
                typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
                for (boost::tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
                    if (pinning[*s_ai] != num_vertices(G)) {
                        typename boost::graph_traits<Graph>::edge_descriptor e;
                        bool is_e;
                        boost::tie(e, is_e) = edge(g, pinning[*s_ai], G);
                        if (!is_e || !edge_equal( edge(s, *s_ai, S).first , e, S, G) )
                            return;
                    }
                visited[g] = true;
                pinning[s] = g;
                if (visited.count() < num_vertices(S)) {
                    typename boost::graph_traits<Graph>::adjacency_iterator g_ai, g_ae;
                    for (boost::tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
                        if (!placed[*s_ai]) {
                            placed[*s_ai] = true;
                            stack.push_back(std::make_pair(*s_ai, g));
                        }
                    SubgraphVertex t = stack[0].first;
                    boost::tie(g_ai, g_ae) = adjacent_vertices(stack[0].second, G);
                    stack.pop_front();
                    for (; g_ai != g_ae; ++g_ai)
                        if (!visited[*g_ai])
                            detail::lattice_constant_walker(
                                  t
                                , *g_ai
                                , S
                                , G
                                , I
                                , matches
                                , distance_to_boarder
                                , stack
                                , placed
                                , visited
                                , pinning
                                , subgraph_orbit
                                , unit_cell_size
                                , geometric_info
                                , breaking_vertex
                                , vertex_equal
                                , edge_equal
                                , exit_on_match
                            );
                } else
                    lattice_constant_insert<Subgraph, Graph, 20, 2>(
                          S
                        , G
                        , I
                        , matches
                        , distance_to_boarder
                        , pinning
                        , subgraph_orbit
                        , unit_cell_size
                        , geometric_info
                        , breaking_vertex
                        , exit_on_match
                    );
                pinning[s] = num_vertices(G);
                visited[g] = false;
            }

            // Input: Subgraph, Graph, vertices of G contained in mapping of S on G
            // Output: lattice_constant of S in G containing v
            template<typename Subgraph, typename Graph, typename GeometricInfo, typename BreakingVertex, typename VertexEqual, typename EdgeEqual, typename ExitOnMatch> std::size_t lattice_constant_impl(
                  Subgraph const & S
                , Graph const & G
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & V
                , std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
                , typename partition_type<Subgraph>::type const & subgraph_orbit
                , std::size_t unit_cell_size
                , GeometricInfo & geometric_info
                , BreakingVertex const & breaking_vertex
                , VertexEqual & vertex_equal
                , EdgeEqual & edge_equal
                , ExitOnMatch exit_on_match
            ) {
                // Assume the vertex desciptor is an unsigned integer type (since we want to use it as an index for a vector)
                BOOST_STATIC_ASSERT((boost::is_unsigned<typename alps::graph_traits<Subgraph>::vertex_descriptor>::value));
                assert(num_vertices(S) > 0);
                // if larger, extend the space
                assert(num_vertices(S) < 21);
                assert(num_edges(S) < 21);

                BOOST_STATIC_ASSERT((boost::is_unsigned<typename alps::graph_traits<Graph>::vertex_descriptor>::value));
                assert(num_vertices(G) > 0);

                // make sure, that a distance in one direction fits in a boost::uint8_t
                assert(num_vertices(G) < 256 * 256);

                // If the lattice has more than 2 dimensions improve lattice_constant_insert
                assert(distance_to_boarder.size() < 3);

                // orbit index => vertices
                std::vector<std::size_t> I(num_vertices(S));
                // Io = {(mi, j) : ni element of Vj
                for (typename partition_type<Subgraph>::type::const_iterator it = subgraph_orbit.begin(); it != subgraph_orbit.end(); ++it)
                    for (typename partition_type<Subgraph>::type::value_type::const_iterator jt = it->begin(); jt != it->end(); ++jt)
                        I[*jt] = it - subgraph_orbit.begin();

                // Matched embeddings
                boost::unordered_set<embedding_generic_type> matches;

                for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = V.begin(); it != V.end(); ++it)
                    for (typename partition_type<Subgraph>::type::const_iterator jt = subgraph_orbit.begin(); jt != subgraph_orbit.end(); ++jt)
                        if (out_degree(jt->front(), S) <= out_degree(*it, G)) {
                            // TODO: shouldn't out_degree be just degree?
                            // TODO: use dynamicbitset
                            boost::dynamic_bitset<> placed(num_vertices(S));
                            boost::dynamic_bitset<> visited(num_vertices(G));
                            std::deque<std::pair<
                                  typename boost::graph_traits<Subgraph>::vertex_descriptor
                                , typename boost::graph_traits<Graph>::vertex_descriptor
                            > > stack;
                            std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> pinning(num_vertices(S), num_vertices(G));
                            placed[jt->front()] = true;
                            lattice_constant_walker(
                                  jt->front()
                                , *it
                                , S
                                , G
                                , I
                                , matches
                                , distance_to_boarder
                                , stack
                                , placed
                                , visited
                                , pinning
                                , subgraph_orbit
                                , unit_cell_size
                                , geometric_info
                                , breaking_vertex
                                , vertex_equal
                                , edge_equal
                                , exit_on_match
                            );
                            break;
                        }
                return matches.size();
            }

        } // end namespace detail
    } // end namespace graph
} // end namespace alps

#endif // ALPS_GRAPH_DETAIL_LATTICE_CONSTANT_IMPL_HPP
