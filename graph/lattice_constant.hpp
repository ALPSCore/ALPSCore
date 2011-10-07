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

#include <alps/graph/canonical_properties.hpp>

namespace alps {
	namespace graph {
	
		namespace detail {
		
			template<typename Subgraph, typename Graph> void lattice_constant_walker(
				  typename boost::graph_traits<Graph>::vertex_descriptor const & s
				, typename boost::graph_traits<Graph>::vertex_descriptor const & g
				, Subgraph const & S
				, Graph const & G
				, std::map<typename boost::graph_traits<Graph>::vertex_descriptor, std::size_t> & I
				, std::set<std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > > & matches
				, std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > match
				, std::set<typename boost::graph_traits<Subgraph>::vertex_descriptor> placed
				, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> visited
			) {
				if (out_degree(s, S) > out_degree(g, G))
					return;
				// TODO: check colored graphs ...
				placed.insert(s);
				visited.insert(g);
				if (placed.size() == num_vertices(S))
					matches.insert(match);
				else {
					if (match.find(I[s]) == match.end())
						match[I[s]] = std::set<typename boost::graph_traits<Graph>::vertex_descriptor>();
					match[I[s]].insert(g);
					typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
					typename boost::graph_traits<Graph>::adjacency_iterator g_ai, g_ae;
					for (tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
						if (placed.find(*s_ai) == placed.end())
							for (tie(g_ai, g_ae) = adjacent_vertices(g, G); g_ai != g_ae; ++g_ai)
								if (visited.find(*g_ai) == visited.end())
									detail::lattice_constant_walker(*s_ai, *g_ai, S, G, I, matches, match, placed, visited);
				}
			}
		}

		// Input: Subgraph, Graph, vertices of G contained in mapping of S on G
		// Output: lattice_constant of S in G containing v
		template<typename Subgraph, typename Graph> std::size_t lattice_constant(
			  Subgraph const & S
			, Graph const & G
			, std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & v
		) {
			if (v.size() != 1)
				ALPS_NGS_THROW_RUNTIME_ERROR("not Impl!")
			// orbit index => vertices
			std::set<std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > > matches;
			typename partition_type<Graph>::type orbit = boost::get(canonical_properties(S));
			std::map<typename boost::graph_traits<Subgraph>::vertex_descriptor, std::size_t> I;
			// Io = {(mi, j) : ni element of Vj
			detail::partition_indeces(I, orbit, S);
			for (typename partition_type<Graph>::type::const_iterator it = orbit.begin(); it != orbit.end(); ++it) {
				std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > match;
				std::set<typename boost::graph_traits<Subgraph>::vertex_descriptor> placed;
				std::set<typename boost::graph_traits<Graph>::vertex_descriptor> visited;
				detail::lattice_constant_walker(it->front(), v.front(), S, G, I, matches, match, placed, visited);
			}
			return matches.size();
		}
	}
}

#endif
