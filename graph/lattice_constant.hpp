/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_GRAPH_LATTICE_CONSTANT
#define ALPS_GRAPH_LATTICE_CONSTANT

#include <alps/ngs/macros.hpp>

#include <alps/lattice/graph_helper.h>
#include <alps/lattice/graphproperties.h>
#include <alps/numeric/vector_functions.hpp>
#include <alps/graph/canonical_properties.hpp>

#include <boost/array.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include <deque>
#include <vector>
#include <cstring>
#include <algorithm>

namespace alps {
	namespace graph {
	
		namespace detail {
		
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
			template<boost::uint16_t bits> class compressed_set {
				public:
					compressed_set()
						: prefix(10) // take somthing around 16
						// do we really want to take 8 bits ore are 4 enoght
						, mem_size((bits - prefix + 8) * (0x01 << prefix - 3))
						, mem(new boost::uint64_t[mem_size])
					{
					
//					std::cout << prefix << " " << bits << " " << mem_size << " " << boost::uint64_t(std::ceil((bits - prefix) / 8.)) << " " << (0x01 << prefix) << std::endl;
					
					}
				
					~compressed_set() {
						delete[] mem;
					}
					
					bool insert(boost::array<boost::uint8_t, bits / 8> const & data) {
						boost::uint64_t value;
						std::memcpy(data.c_array(), &value, data.size());

						boost::uint8_t ptr = memget<8>(value & ((0x01 << prefix) - 1));
						boost::uint64_t suffix = memget<64>((value & ((0x01 << prefix) - 1)) + 8);

					}

				private:

					compressed_set(compressed_set const &) {}

					template<boost::uint64_t size> inline boost::uint64_t memget(boost::uint64_t index) {
						boost::uint64_t giant = index * ((bits - prefix + 8) >> 6);
						boost::uint64_t baby = index & 0x7F;
						if (index + bits < 64)
							return (mem[giant] >> baby) & ((0x01 << baby) - 1);
						else
							return (mem[giant] >> baby) & ((0x01 << baby) - 1) + ((mem[giant + 1] & ((0x01 << (64 - baby)) - 1)) << 64 - baby);
					}

					// #bits of prefix
					boost::uint16_t prefix;
					// size of mem
					boost::uint64_t mem_size;
					// data are stored here: 8 bits for relative ptr, rest for data
					boost::uint64_t * mem;
			};

			struct embedding_2d_type {
				embedding_2d_type() {
					std::memset(data.c_array(), 0, data.size());
				}
				
				embedding_2d_type(embedding_2d_type const & rhs) {
					std::memcpy(data.c_array(), &(rhs.data[0]), data.size());
				}

				bool operator == (embedding_2d_type const & rhs) const {
					return !memcmp(&(data[0]), &(rhs.data[0]), data.size());
				}
				boost::array<boost::uint8_t, 6> data;
			};

			std::size_t hash_value(embedding_2d_type const & value) {
				using boost::hash_range;
				return hash_range(value.data.begin(), value.data.end());
			}
#endif

#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
			struct embedding_generic_type {

				embedding_generic_type(std::size_t vertices_size, std::size_t edges_size)
					: hash(0)
					, counter(new boost::uint8_t(1))
					, vertices(new std::vector<std::vector<boost::uint16_t> >(vertices_size))
					//warning: & has lower precedence than ==; == will be evaluated first [-Wparentheses]
					, edges(new std::vector<boost::uint64_t>((edges_size >> 6) + (edges_size & 0x7F == 0 ? 0 : 1)))
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
#endif

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
						distance_to_boarder[d][v = *it] = 0;
						if (v != num_vertices(graph))
							while ((v = translations[d][v]) != num_vertices(graph))
								++distance_to_boarder[d][*it];
					}
				}
			}
			
			struct embedding_found {};

			// TODO: move back into main function after optimizing
			template<typename Subgraph, typename Graph> void lattice_constant_insert(
				  Subgraph const & S
				, Graph const & G
				, std::vector<std::size_t> const & I
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
				, boost::unordered_set<embedding_2d_type> & matches_2d
#endif
#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
				, boost::unordered_set<embedding_generic_type> & matches_generic
#endif
				, std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
				, std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
				, typename partition_type<Subgraph>::type const & subgraph_orbit
				, boost::mpl::true_
			) {
				throw embedding_found();
			}

			// TODO: move back into main function after optimizing
			template<typename Subgraph, typename Graph> void lattice_constant_insert(
				  Subgraph const & S
				, Graph const & G
				, std::vector<std::size_t> const & I
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
				, boost::unordered_set<embedding_2d_type> & matches_2d
#endif
#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
				, boost::unordered_set<embedding_generic_type> & matches_generic
#endif
				, std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
				, std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & pinning
				, typename partition_type<Subgraph>::type const & subgraph_orbit
				, boost::mpl::false_
			) {
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)

				embedding_2d_type embedding_2d;
				// if S has more than 20 vertices, chane this
				boost::array<boost::uint8_t, 20> ordred_vertices;
				std::memset(ordred_vertices.c_array(), boost::uint8_t(num_vertices(S)), ordred_vertices.size());
				boost::array<boost::uint8_t, 20> vertices_order;
				boost::array<boost::uint_t<32>::fast, 20> ordred_edges;
				std::memset(ordred_edges.c_array(), 0, num_edges(S));
				// assume the lattice has only 2 axis
				boost::array<boost::uint_t<8>::fast, 2> min_dist_to_boarder = { {
					  boost::integer_traits<boost::uint8_t>::const_max - 1
					, boost::integer_traits<boost::uint8_t>::const_max - 1
				} };
				// take offset to bottom of node most left
				embedding_2d.data[5] = boost::integer_traits<boost::uint8_t>::const_max - 1;

				// If the lattice has more than two dimensions, change that
				for (boost::uint_t<8>::fast i = 0; i < pinning.size(); ++i) {
					// TODO: min_dist_to_boarder is not really used ...REMOVE IT!
					min_dist_to_boarder[0] = std::min(min_dist_to_boarder[0], distance_to_boarder[0][pinning[i]]);
					min_dist_to_boarder[1] = std::min(min_dist_to_boarder[1], distance_to_boarder[1][pinning[i]]);
					// TODO: this can be done faster ...
					boost::uint_t<8>::fast index = 0, vertex = i;
					// TODO: make a lookup table for that and pass the reference at each call ...
					for (std::size_t i = 0; i < I[vertex]; ++i)
						index += subgraph_orbit[i].size();
					for (; ordred_vertices[index] != boost::uint8_t(num_vertices(S)); ++index)
						if (
							   distance_to_boarder[0][pinning[vertex]] < distance_to_boarder[0][pinning[ordred_vertices[index]]]
							or (
								    distance_to_boarder[0][pinning[vertex]] == distance_to_boarder[0][pinning[ordred_vertices[index]]]
								and distance_to_boarder[1][pinning[vertex]] < distance_to_boarder[1][pinning[ordred_vertices[index]]]
							)
						)
							std::swap(vertex, ordred_vertices[index]);
					ordred_vertices[index] = vertex;
				}
				for (boost::uint_t<8>::fast i = 0; i < pinning.size(); ++i)
					if (distance_to_boarder[0][pinning[i]] == min_dist_to_boarder[0])
						embedding_2d.data[5] = std::min(embedding_2d.data[5], distance_to_boarder[1][pinning[i]]);
				embedding_2d.data[5] -= min_dist_to_boarder[1];
				
				for (boost::uint_t<8>::fast i = 0; i < ordred_vertices.size(); ++i)
					vertices_order[ordred_vertices[i]] = i;

				std::size_t pos = 0;
				typename boost::graph_traits<Subgraph>::edge_iterator ei, ee;
				for (boost::tie(ei, ee) = edges(S); ei != ee; ++ei, ++pos) {
					boost::uint_t<8>::fast vs = source(*ei, S), vt = target(*ei, S);
					if (vertices_order[vs] < vertices_order[vt])
						std::swap(vs, vt);
					ordred_edges[pos] = (vertices_order[vs] << 16) + (vertices_order[vt] << 8) + (distance_to_boarder[0][pinning[vs]] != distance_to_boarder[0][pinning[vt]]
						? (distance_to_boarder[0][pinning[vs]] < distance_to_boarder[0][pinning[vt]] ? 0x00 : 0x01)
						: (distance_to_boarder[1][pinning[vs]] < distance_to_boarder[1][pinning[vt]] ? 0x02 : 0x03)
					);
				}
				std::sort(ordred_edges.begin(), ordred_edges.begin() + num_edges(S));
				for (std::size_t i = 0; i < num_edges(S); ++i)
					embedding_2d.data[i >> 2] |= (ordred_edges[i] & 0x7F) << ((i << 1) & 0x07);

#ifndef CHECK_COMPRESSED_EMBEDDING
				matches_2d.insert(embedding_2d);
#endif

#endif
#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)

				embedding_generic_type embedding_generic(subgraph_orbit.size(), num_vertices(S) * (num_vertices(S) + 1) / 2);

				for (std::vector<std::vector<boost::uint16_t> >::iterator it = embedding_generic.vertices->begin(); it != embedding_generic.vertices->end(); ++it)
					it->reserve(subgraph_orbit[it - embedding_generic.vertices->begin()].size());

				std::size_t bits_per_dim = 0;
				while ((0x01 << ++bits_per_dim) < num_vertices(S));
				assert((0x01 << (distance_to_boarder.size() * bits_per_dim)) < boost::integer_traits<boost::uint16_t>::const_max);

				std::vector<boost::uint_t<8>::fast> distances(distance_to_boarder.size(), num_vertices(G));
				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it)
					for(std::size_t d = 0; d < distance_to_boarder.size(); ++d)
						distances[d] = std::min(distances[d], distance_to_boarder[d][*it]);

				std::vector<boost::uint16_t> lattice_pinning(pinning.size());
				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it) {
					lattice_pinning[it - pinning.begin()] = distance_to_boarder[0][*it] - distances[0];
					for(std::size_t d = 1; d < distance_to_boarder.size(); ++d) {
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
					(*embedding_generic.edges)[index >> 6] |= 0x01 << (index & 0x7F);
				}
				for (std::vector<boost::uint64_t>::const_iterator it = embedding_generic.edges->begin(); it != embedding_generic.edges->end(); ++it) {
					using boost::hash_combine;
					hash_combine(embedding_generic.hash, *it);
				}
#ifndef CHECK_COMPRESSED_EMBEDDING
				matches_generic.insert(embedding_generic);
#endif

#endif

#ifdef CHECK_COMPRESSED_EMBEDDING
				boost::unordered_set<embedding_2d_type>::const_iterator it_2d;
				boost::unordered_set<embedding_generic_type>::const_iterator it_generic;
				bool b_2d, b_generic;
				boost::tie(it_2d, b_2d) = matches_2d.insert(embedding_2d);
				boost::tie(it_generic, b_generic) = matches_generic.insert(embedding_generic);
				
				if (b_2d and !b_generic)
					std::cout << "match on compressed, not in generic";
				
				if (!b_2d and b_generic) {
					// TODO: also output the value stored in it_2d to see the other part ...
					for (std::size_t i = 0; i < num_edges(S); ++i) {
						std::cout << ((ordred_edges[i] >> 16) & 0x7F) << "-" << ((ordred_edges[i] >> 8) & 0x7F) << " ";
						switch (ordred_edges[i] & 0x7F) {
							case 0: std::cout << "->"; break;
							case 1: std::cout << "<-"; break;
							case 2: std::cout << "`|`"; break;
							case 3: std::cout << ".|."; break;
						}
						std::cout << std::endl;
					}
					std::cout << ":" << unsigned(embedding_2d.data[5]) << std::endl << std::endl;
				}
#endif

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
			
/*

//				std::vector<boost::array<boost::uint16_t, 3> > ordred_edges(num_edges(S));
//				std::size_t pos = 0;
				typename boost::graph_traits<Subgraph>::edge_iterator s_ei, s_ee;
				for (boost::tie(s_ei, s_ee) = edges(S); s_ei != s_ee; ++s_ei, ++pos) {
					std::size_t v1 = std::min(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
					std::size_t v2 = std::max(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
					std::size_t index = v1 * num_vertices(S) - (v1 - 1) * v1 / 2 + v2 - v1;
					(*embedding.edges)[index >> 6] |= 0x01 << (index & 0x7F);
/ *					
					typename boost::graph_traits<Subgraph>::vertex_descriptor vs = source(*s_ei, S);
					typename boost::graph_traits<Subgraph>::vertex_descriptor vt = target(*s_ei, S);
					if (lattice_pinning[source(*s_ei, S)] > lattice_pinning[target(*s_ei, S)])
						std::swap(vs, vt);
					ordred_edges[pos][0] = lattice_pinning[vs];
					ordred_edges[pos][1] = lattice_pinning[vt];
					ordred_edges[pos][2] = distance_to_boarder[0][pinning[vs]] != distance_to_boarder[0][pinning[vt]]
						? (distance_to_boarder[0][pinning[vs]] < distance_to_boarder[0][pinning[vt]] ? 0x00 : 0x01)
						: (distance_to_boarder[1][pinning[vs]] < distance_to_boarder[1][pinning[vt]] ? 0x02 : 0x03)
					;
* /
				}
				for (std::vector<boost::uint64_t>::const_iterator it = embedding.edges->begin(); it != embedding.edges->end(); ++it) {
					using boost::hash_combine;
					hash_combine(embedding.hash, *it);
				}
/ *
				std::sort(ordred_edges.begin(), ordred_edges.end());
				for (std::size_t i = 0; i < ordred_edges.size(); ++i)
					embedding.data[i >> 2] |= ordred_edges[i][2] << ((i << 1) & 0x07);
* /


				embedding_type embedding;
				std::memset(embedding.c_array(), 0, embedding.size());

				std::vector<std::vector<boost::uint16_t> > vertices(subgraph_orbit.size());
				for (std::vector<std::vector<boost::uint16_t> >::iterator it = vertices.begin(); it != vertices.end(); ++it)
					it->reserve(subgraph_orbit[it - vertices.begin()].size());

				// TODO: move this calculation to the main function
				std::size_t bits_per_dim = 0;
				while ((0x01 << ++bits_per_dim) < num_vertices(S));
				assert((0x01 << (distance_to_boarder.size() * bits_per_dim)) < boost::integer_traits<boost::uint16_t>::const_max);
				
				std::vector<boost::uint_t<8>::fast> distances(distance_to_boarder.size(), num_vertices(G));
				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it)
					for(std::size_t d = 0; d < distance_to_boarder.size(); ++d)
						distances[d] = std::min(distances[d], distance_to_boarder[d][*it]);

				// TODO: embedding.vertices can be created in a sorted way
				std::vector<boost::uint16_t> lattice_pinning(pinning.size())run;
				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it) {
					lattice_pinning[it - pinning.begin()] = distance_to_boarder[0][*it] - distances[0];
					for(std::size_t d = 1; d < distance_to_boarder.size(); ++d) {
						lattice_pinning[it - pinning.begin()] <<= bits_per_dim;
						lattice_pinning[it - pinning.begin()] += distance_to_boarder[d][*it] - distances[d];
					}
					vertices[I[it - pinning.begin()]].push_back(lattice_pinning[it - pinning.begin()]);
				}

				for (std::vector<std::vector<boost::uint16_t> >::iterator it = vertices.begin(); it != vertices.end(); ++it)
					std::sort(it->begin(), it->end());

				for (std::vector<boost::uint16_t>::iterator it = lattice_pinning.begin(); it != lattice_pinning.end(); ++it) {
					std::vector<boost::uint16_t>::iterator jt = vertices[I[it - lattice_pinning.begin()]].begin();
					for (; *jt != *it; ++jt);
					*it = jt - vertices[I[it - lattice_pinning.begin()]].begin();
					for (std::size_t i = 0; i < I[it - lattice_pinning.begin()]; ++i)
						*it += vertices[i].size();
				}

				std::vector<boost::array<boost::uint16_t, 3> > ordred_edges(num_edges(S));
				std::size_t pos = 0;
				typename boost::graph_traits<Subgraph>::edge_iterator s_ei, s_ee;
				for (boost::tie(s_ei, s_ee) = edges(S); s_ei != s_ee; ++s_ei, ++pos) {
					typename boost::graph_traits<Subgraph>::vertex_descriptor vs = source(*s_ei, S);
					typename boost::graph_traits<Subgraph>::vertex_descriptor vt = target(*s_ei, S);
					
					ordred_edges[pos][0] = std::min(lattice_pinning[vs], lattice_pinning[vt]);
					ordred_edges[pos][1] = std::max(lattice_pinning[vs], lattice_pinning[vt]);
					ordred_edges[pos][2] = distance_to_boarder[0][pinning[vs]] != distance_to_boarder[0][pinning[vt]]
						? (distance_to_boarder[0][pinning[vs]] < distance_to_boarder[0][pinning[vt]] ? 0x00 : 0x01)
						: (distance_to_boarder[1][pinning[vs]] < distance_to_boarder[1][pinning[vt]] ? 0x02 : 0x03)
					;
				}
				std::sort(ordred_edges.begin(), ordred_edges.end());
				
				for (std::size_t i = 0; i < ordred_edges.size(); ++i)
					embedding[i >> 2] |= ordred_edges[i][2] << ((i << 1) & 0x07);
				
				
				
				
				/*
				
				std::memset(embedding.c_array(), 0, embedding.size());
				
				boost::array<boost::uint_t<8>::fast, 2> min_dist_to_boarder = { {
					  boost::integer_traits<boost::uint8_t>::const_max - 1
					, boost::integer_traits<boost::uint8_t>::const_max - 1
				} };

				std::size_t pos = 0;
				typename boost::graph_traits<Subgraph>::edge_iterator et, ee;
				for (tie(et, ee) = edges(S); et != ee; ++et, pos += 2) {
					typename boost::graph_traits<Subgraph>::vertex_descriptor vs = source(*et, S);
					typename boost::graph_traits<Subgraph>::vertex_descriptor vt = target(*et, S);

					min_dist_to_boarder[0] = std::min(min_dist_to_boarder[0], std::min(distance_to_boarder[0][pinning[vs]], distance_to_boarder[0][pinning[vt]]));
					min_dist_to_boarder[1] = std::min(min_dist_to_boarder[1], std::min(distance_to_boarder[1][pinning[vs]], distance_to_boarder[1][pinning[vt]]));
					
					boost::uint8_t mask = distance_to_boarder[0][pinning[vs]] != distance_to_boarder[0][pinning[vt]]
						? (distance_to_boarder[0][pinning[vs]] < distance_to_boarder[0][pinning[vt]] ? 0x00 : 0x01)
						: (distance_to_boarder[1][pinning[vs]] < distance_to_boarder[1][pinning[vt]] ? 0x02 : 0x03)
					;

					embedding[pos >> 3] |= mask << (pos & 0x07);
					
					std::cout << unsigned(mask) << " | ";
					
				}
				
				std::cout << unsigned(embedding[0]) << " " << unsigned(min_dist_to_boarder[1]) << " | " << unsigned(min_dist_to_boarder[0]) << std::endl << ">";
				
				embedding[0] -= min_dist_to_boarder[1];
				
				
				for (std::size_t i = 0; i < embedding.size(); ++i)
					std::cout << unsigned(embedding[i]) << " ";
				std::cout << std::endl;

				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it)
					std::cout << (it - pinning.begin()) << ":" << *it << " ";
				std::cout << std::endl;


/*

				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it) {
					if (distance_to_boarder[0][*it] < min_dist_to_boarder[0] or (distance_to_boarder[0][*it] == min_dist_to_boarder[0] and distance_to_boarder[1][*it] < embedding[0]))
						embedding.data[0] = distance_to_boarder[1][*it];
					min_dist_to_boarder[0] = std::min(min_dist_to_boarder[0], distance_to_boarder[0][*it]);
					min_dist_to_boarder[1] = std::min(min_dist_to_boarder[1], distance_to_boarder[1][*it]);

					// TODO: this can be done faster ...
					boost::uint8_t index = 0, vertex = it - pinning.begin();
					// TODO: make a lookup table for that and pass the reference at each call ...
					for (std::size_t i = 0; i < I[vertex]; ++i)
						index += subgraph_orbit[i].size();
						
std::cout << int(index) << " " << ordred_vertices.size() << " " << int(ordred_vertices[index]) << std::endl;
						
					for (; ordred_vertices[index] != num_vertices(S); ++index)
						if (
							   distance_to_boarder[pinning[vertex]][0] < distance_to_boarder[pinning[ordred_vertices[index]]][0]
							or (
								    distance_to_boarder[pinning[vertex]][0] == distance_to_boarder[pinning[ordred_vertices[index]]][0]
								and distance_to_boarder[pinning[vertex]][1] < distance_to_boarder[pinning[ordred_vertices[index]]][1]
							)
						)
							std::swap(vertex, ordred_vertices[index]);
					ordred_vertices[index] = vertex;
				}
				embedding.data[0] -= min_dist_to_boarder[1];





/*				// if S has more than 28 vertices, chane this
				boost::array<boost::uint8_t, 28> ordred_vertices;
				std::memset(ordred_vertices.c_array(), num_vertices(S), 28);
				boost::array<boost::uint8_t, 28> vertices_order;
				// assume the lattice has only 2 axis
				boost::array<boost::uint_t<16>::fast, 2> min_dist_to_boarder = { { num_vertices(G), num_vertices(G) } };
				// most_left_vertex on dimension 0 is always zero, take the first 8 bits for the second dimensions of the most left vertex
				embedding.data[0] = num_vertices(G);
				// initialize the seond representation container
				embedding.data[1] = 0;

				// If the lattice has more than two dimensions, change that
				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it) {
					if (distance_to_boarder[0][*it] < min_dist_to_boarder[0] or (distance_to_boarder[0][*it] == min_dist_to_boarder[0] and distance_to_boarder[1][*it] < embedding.data[0]))
						embedding.data[0] = distance_to_boarder[1][*it];
					min_dist_to_boarder[0] = std::min(min_dist_to_boarder[0], distance_to_boarder[0][*it]);
					min_dist_to_boarder[1] = std::min(min_dist_to_boarder[1], distance_to_boarder[1][*it]);

					// TODO: this can be done faster ...
					boost::uint8_t index = 0, vertex = it - pinning.begin();
					// TODO: make a lookup table for that and pass the reference at each call ...
					for (std::size_t i = 0; i < I[vertex]; ++i)
						index += subgraph_orbit[i].size();
						
std::cout << int(index) << " " << ordred_vertices.size() << " " << int(ordred_vertices[index]) << std::endl;
						
					for (; ordred_vertices[index] != num_vertices(S); ++index)
						if (
							   distance_to_boarder[pinning[vertex]][0] < distance_to_boarder[pinning[ordred_vertices[index]]][0]
							or (
								    distance_to_boarder[pinning[vertex]][0] == distance_to_boarder[pinning[ordred_vertices[index]]][0]
								and distance_to_boarder[pinning[vertex]][1] < distance_to_boarder[pinning[ordred_vertices[index]]][1]
							)
						)
							std::swap(vertex, ordred_vertices[index]);
					ordred_vertices[index] = vertex;
				}
				embedding.data[0] -= min_dist_to_boarder[1];
				for (std::size_t i = 0; i < ordred_vertices.size(); ++i)
					vertices_order[ordred_vertices[i]] = i;

				typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
				for (boost::array<boost::uint8_t, 28>::const_iterator it = ordred_vertices.begin(); it != ordred_vertices.end(); ++it)
					for (tie(s_ai, s_ae) = adjacent_vertices(it - ordred_vertices.begin(), S); s_ai != s_ae; ++s_ai) {
						boost::uint8_t pos = 5 + 4 * (it - ordred_vertices.begin()) + (distance_to_boarder[0][it - ordred_vertices.begin()] == distance_to_boarder[0][*s_ai]
							? (distance_to_boarder[0][it - ordred_vertices.begin()] - distance_to_boarder[0][*s_ai] < 0 ? 0 : 1)
							: (distance_to_boarder[1][it - ordred_vertices.begin()] - distance_to_boarder[1][*s_ai] < 0 ? 2 : 3)
						);
						embedding.data[pos >> 6] |= 0x01 << (pos & 0x7F);
					}
/*
				typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;



				typename boost::graph_traits<Subgraph>::edge_iterator s_ei, s_ee;
				for (boost::tie(s_ei, s_ee) = edges(S); s_ei != s_ee; ++s_ei) {
					boost::uint8_t v1 = vertices_order[source(*s_ei, S)]
				
					boost::uint8_t = find(ordred_vertices.begin(), ordred_vertices.end()) - ordred_vertices.end();
				
					std::size_t v1 = std::min(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
					std::size_t v2 = std::max(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
					std::size_t index = v1 * num_vertices(S) - (v1 - 1) * v1 / 2 + v2 - v1;
					(*embedding.edges)[index >> 8] |= 0x01 << (index & 0x7F);
				}

embedding.data[0]
				

				// vertex i, dim j, sign of dim k: 5 + 4 * i + 2 * j + (k < 0 ? -1 : 1)

				
				
/*
				typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
				for (tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
					if (pinning[*s_ai] != num_vertices(G)) {
						typename boost::graph_traits<Graph>::edge_descriptor e;
						bool is_e;
						tie(e, is_e) = edge(g, pinning[*s_ai], G);
						if (!is_e or !lattice_constant_edge_equal(
							  edge(s, *s_ai, S).first
							, e
							, S
							, G
							, typename detail::has_coloring<typename boost::edge_property_type<Graph>::type>::type())
						)
							return;
					}
*/
				
				
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				/*
				embedding_type embedding(subgraph_orbit.size(), num_vertices(S) * (num_vertices(S) + 1) / 2);
				for (std::vector<std::vector<boost::uint16_t> >::iterator it = embedding.vertices->begin(); it != embedding.vertices->end(); ++it)
					it->reserve(subgraph_orbit[it - embedding.vertices->begin()].size());

				// TODO: move this calculation to the main function
				std::size_t bits_per_dim = 0;
				while ((0x01 << ++bits_per_dim) < num_vertices(S));
				assert((0x01 << (distance_to_boarder.size() * bits_per_dim)) < boost::integer_traits<boost::uint16_t>::const_max);

				std::vector<boost::uint_t<8>::fast> distances(distance_to_boarder.size(), num_vertices(G));
				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it)
					for(std::size_t d = 0; d < distance_to_boarder.size(); ++d)
						distances[d] = std::min(distances[d], distance_to_boarder[d][*it]);

				// TODO: embedding.vertices can be created in a sorted way
				std::vector<boost::uint16_t> lattice_pinning(pinning.size());
				for (typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it = pinning.begin(); it != pinning.end(); ++it) {
					lattice_pinning[it - pinning.begin()] = distance_to_boarder[0][*it] - distances[0];
					for(std::size_t d = 1; d < distance_to_boarder.size(); ++d) {
						lattice_pinning[it - pinning.begin()] <<= bits_per_dim;
						lattice_pinning[it - pinning.begin()] += distance_to_boarder[d][*it] - distances[d];
					}
					(*embedding.vertices)[I[it - pinning.begin()]].push_back(lattice_pinning[it - pinning.begin()]);
				}

				for (std::vector<std::vector<boost::uint16_t> >::iterator it = embedding.vertices->begin(); it != embedding.vertices->end(); ++it) {
					using boost::hash_combine;
					std::sort(it->begin(), it->end());
					for (std::vector<boost::uint16_t>::const_iterator jt = it->begin(); jt != it->end(); ++jt)
						hash_combine(embedding.hash, *jt);
				}

				for (std::vector<boost::uint16_t>::iterator it = lattice_pinning.begin(); it != lattice_pinning.end(); ++it) {
					std::vector<boost::uint16_t>::iterator jt = (*embedding.vertices)[I[it - lattice_pinning.begin()]].begin();
					for (; *jt != *it; ++jt);
					*it = jt - (*embedding.vertices)[I[it - lattice_pinning.begin()]].begin();
					for (std::size_t i = 0; i < I[it - lattice_pinning.begin()]; ++i)
						*it += (*embedding.vertices)[i].size();
				}

				typename boost::graph_traits<Subgraph>::edge_iterator s_ei, s_ee;
				for (boost::tie(s_ei, s_ee) = edges(S); s_ei != s_ee; ++s_ei) {
					std::size_t v1 = std::min(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
					std::size_t v2 = std::max(lattice_pinning[source(*s_ei, S)], lattice_pinning[target(*s_ei, S)]);
					std::size_t index = v1 * num_vertices(S) - (v1 - 1) * v1 / 2 + v2 - v1;
					(*embedding.edges)[index >> 6] |= 0x01 << (index & 0x7F);
				}
				for (std::vector<boost::uint64_t>::const_iterator it = embedding.edges->begin(); it != embedding.edges->end(); ++it) {
					using boost::hash_combine;
					hash_combine(embedding.hash, *it);
				}

				matches.insert(embedding);
				*/
			}

			template<typename Subgraph, typename Graph> bool lattice_constant_vertex_equal(
				  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s
				, typename boost::graph_traits<Graph>::vertex_descriptor const & g
				, Subgraph const & S
				, Graph const & G
				, boost::mpl::true_
			) {
				return get(boost::vertex_name_t(), S)[s] == get(boost::vertex_name_t(), G)[g];
			} 

			template<typename Subgraph, typename Graph> bool lattice_constant_edge_equal(
				  typename boost::graph_traits<Subgraph>::edge_descriptor const & s_e
				, typename boost::graph_traits<Graph>::edge_descriptor const & g_e
				, Subgraph const & S
				, Graph const & G
				, boost::mpl::true_
			) {
				return get(alps::edge_type_t(), S)[s_e] == get(alps::edge_type_t(), G)[g_e];
			}

			template<typename Subgraph, typename Graph> bool lattice_constant_vertex_equal(
				  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s
				, typename boost::graph_traits<Graph>::vertex_descriptor const & g
				, Subgraph const & S
				, Graph const & G
				, boost::mpl::false_
			) {
				return true;
			}

			template<typename Subgraph, typename Graph> bool lattice_constant_edge_equal(
				  typename boost::graph_traits<Subgraph>::edge_descriptor const & s_e
				, typename boost::graph_traits<Graph>::edge_descriptor const & g_e
				, Subgraph const & S
				, Graph const & G
				, boost::mpl::false_
			) {
				return true;
			}

			// TODO: make an object out of walker
			template<typename Subgraph, typename Graph, typename ExitOnMatch> void lattice_constant_walker(
				  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s
				, typename boost::graph_traits<Graph>::vertex_descriptor const & g
				, Subgraph const & S
				, Graph const & G
				, std::vector<std::size_t> const & I
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
				, boost::unordered_set<embedding_2d_type> & matches_2d
#endif
#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
				, boost::unordered_set<embedding_generic_type> & matches_generic
#endif
				, std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
				, std::deque<std::pair<
					  typename boost::graph_traits<Subgraph>::vertex_descriptor
					, typename boost::graph_traits<Graph>::vertex_descriptor
				  > > stack
				, boost::dynamic_bitset<> placed
				, boost::dynamic_bitset<> & visited
				, std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> & pinning
				, typename partition_type<Subgraph>::type const & subgraph_orbit
				, ExitOnMatch exit_on_match
			) {
				typedef typename boost::graph_traits<Subgraph>::vertex_descriptor SubgraphVertex;
				typedef typename boost::graph_traits<Graph>::vertex_descriptor GraphVertex;

				if (out_degree(s, S) > out_degree(g, G))
					return;
				if (!lattice_constant_vertex_equal(
					  s
					, g
					, S
					, G
					, typename detail::has_coloring<typename boost::vertex_property_type<Graph>::type>::type())
				)
					return;
				typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
				for (tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
					if (pinning[*s_ai] != num_vertices(G)) {
						typename boost::graph_traits<Graph>::edge_descriptor e;
						bool is_e;
						tie(e, is_e) = edge(g, pinning[*s_ai], G);
						if (!is_e or !lattice_constant_edge_equal(
							  edge(s, *s_ai, S).first
							, e
							, S
							, G
							, typename detail::has_coloring<typename boost::edge_property_type<Graph>::type>::type())
						)
							return;
					}
				visited[g] = true;
				pinning[s] = g;
				if (visited.count() < num_vertices(S)) {
					typename boost::graph_traits<Graph>::adjacency_iterator g_ai, g_ae;
					for (tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
						if (!placed[*s_ai]) {
							placed[*s_ai] = true;
							stack.push_back(std::make_pair(*s_ai, g));
						}
					SubgraphVertex t = stack[0].first;
					tie(g_ai, g_ae) = adjacent_vertices(stack[0].second, G);
					stack.pop_front();
					for (; g_ai != g_ae; ++g_ai)
						if (!visited[*g_ai])
							detail::lattice_constant_walker(
								  t
								, *g_ai
								, S
								, G
								, I
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
								, matches_2d
#endif
#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
								, matches_generic
#endif
								, distance_to_boarder
								, stack
								, placed
								, visited
								, pinning
								, subgraph_orbit
								, exit_on_match
							);
				} else
					lattice_constant_insert(
						  S
						, G
						, I
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
						, matches_2d
#endif
#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
						, matches_generic
#endif
						, distance_to_boarder
						, pinning
						, subgraph_orbit
						, exit_on_match
					);
				pinning[s] = num_vertices(G);
				visited[g] = false;
			}

			// Input: Subgraph, Graph, vertices of G contained in mapping of S on G
			// Output: lattice_constant of S in G containing v
			template<typename Subgraph, typename Graph, typename ExitOnMatch> std::size_t lattice_constant_impl(
				  Subgraph const & S
				, Graph const & G
				, typename boost::graph_traits<Graph>::vertex_descriptor v
				, std::vector<std::vector<boost::uint_t<8>::fast> > const & distance_to_boarder
				, typename partition_type<Subgraph>::type const & subgraph_orbit
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
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)




				// TODO: fixit!
				compressed_set<48> matches_2d_new;





				boost::unordered_set<embedding_2d_type> matches_2d;
#endif
#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
				boost::unordered_set<embedding_generic_type> matches_generic;
#endif

				for (typename partition_type<Subgraph>::type::const_iterator it = subgraph_orbit.begin(); it != subgraph_orbit.end(); ++it)
					if (out_degree(it->front(), S) <= out_degree(v, G)) {
						// TODO: use dynamicbitset
						boost::dynamic_bitset<> placed(num_vertices(S));
						boost::dynamic_bitset<> visited(num_vertices(G));
						std::deque<std::pair<
							  typename boost::graph_traits<Subgraph>::vertex_descriptor
							, typename boost::graph_traits<Graph>::vertex_descriptor
						> > stack;
						std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> pinning(num_vertices(S), num_vertices(G));
						placed[it->front()] = true;
						lattice_constant_walker(
							  it->front()
							, v
							, S
							, G
							, I 
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
							, matches_2d
#endif
#if !defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
							, matches_generic
#endif
							, distance_to_boarder
							, stack
							, placed
							, visited
							, pinning
							, subgraph_orbit
							, exit_on_match
						);
						break;
					}
#if defined(USE_COMPRESSED_EMBEDDING) or defined(CHECK_COMPRESSED_EMBEDDING)
				return matches_2d.size();
#else
				return matches_generic.size();
#endif
			}
		}

		template<typename Subgraph, typename Graph, typename Lattice> std::size_t lattice_constant(
			  Subgraph const & S
			, Graph const & G
			, Lattice const & L
			, typename boost::graph_traits<Graph>::vertex_descriptor v
		) {			
			typedef typename alps::graph_helper<Graph>::lattice_type lattice_type;
			typedef typename alps::lattice_traits<lattice_type>::unit_cell_type::graph_type unit_cell_graph_type;

			// Get the possible translation in the lattice
			std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder(dimension(L), std::vector<boost::uint_t<8>::fast>(num_vertices(G), num_vertices(G)));
			detail::build_translation_table(G, L, distance_to_boarder);
			
			typename partition_type<Subgraph>::type subgraph_orbit = boost::get<2>(canonical_properties(S));

			return detail::lattice_constant_impl(S, G, v, distance_to_boarder, subgraph_orbit, boost::mpl::false_());
		}

		template<typename Subgraph, typename Graph> bool is_embeddable(
			  Subgraph const & S
			, Graph const & G
			, typename boost::graph_traits<Graph>::vertex_descriptor v
			, typename partition_type<Subgraph>::type const & subgraph_orbit			
		) {
			std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder;

			try {
				detail::lattice_constant_impl(S, G, v, distance_to_boarder, subgraph_orbit, boost::mpl::true_());
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
			std::vector<std::vector<boost::uint_t<8>::fast> > distance_to_boarder;

			try {
				typename boost::graph_traits<Graph>::vertex_iterator vt, ve;
				for (boost::tie(vt, ve) = vertices(G); vt != ve; ++vt)
					detail::lattice_constant_impl(S, G, *vt, distance_to_boarder, subgraph_orbit, boost::mpl::true_());
				return false;
			} catch (detail::embedding_found e) {
				return true;
			}
		}		
	}
}
#endif
