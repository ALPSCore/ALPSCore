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

#include <deque>

namespace alps {
	namespace graph {
	
		namespace detail {

			template <typename Graph, typename Lattice> std::vector<
				std::vector<unsigned int> 
			> build_translation_table(Graph const& graph, Lattice const& lattice)
			{
				typedef Lattice lattice_type;
				typedef Graph graph_type;

				typedef typename alps::lattice_traits<lattice_type>::cell_iterator cell_iterator;
				typedef typename alps::lattice_traits<lattice_type>::offset_type offset_type;
				typedef typename alps::lattice_traits<lattice_type>::size_type cell_index_type;
				
				// Assume the vertex desciptor is an unsigned integer type (since we want to use it as an index for a vector)
				BOOST_STATIC_ASSERT((boost::is_unsigned<typename alps::graph_traits<graph_type>::vertex_descriptor>::value));
				assert(num_vertices(graph) > 0);


				std::vector<std::vector<unsigned int> > result(dimension(lattice),std::vector<unsigned int>(num_vertices(graph),num_vertices(graph)));
				unsigned int vtcs_per_ucell = num_vertices(alps::graph::graph(unit_cell(lattice)));
				for(std::size_t d = 0; d < dimension(lattice); ++d)
				{
					for(std::pair<cell_iterator,cell_iterator> c = cells(lattice); c.first != c.second; ++c.first)
					{
						offset_type ofst = offset(*c.first,lattice);
						offset_type move(dimension(lattice));
						move[d] = -1;
						std::pair<bool,bool> on_lattice_pbc_crossing = shift(ofst,move,lattice);
						if(on_lattice_pbc_crossing.first && !on_lattice_pbc_crossing.second)
						{
							const cell_index_type cellidx = index(*c.first,lattice);
							const cell_index_type neighboridx = index(cell(ofst,lattice),lattice);
							for(unsigned int v = 0; v < vtcs_per_ucell; ++v)
								result[d][cellidx*vtcs_per_ucell+v] = neighboridx*vtcs_per_ucell + v;
						}
					}
				}
				return result;
			}

			template<typename Vertex> std::map<unsigned, std::set<Vertex> > lattice_constant_translations(
				  unsigned int invalid
				, std::vector<unsigned int> const & translation
				, std::map<unsigned, std::set<Vertex> > match
			) {
				std::map<unsigned, std::set<Vertex> > moved;
				do {
					moved.clear();
					for (typename std::map<unsigned, std::set<Vertex> >::const_iterator jt = match.begin(); jt != match.end(); ++jt) {
						moved[jt->first] = std::set<Vertex>();
						for (typename std::set<Vertex>::const_iterator kt = jt->second.begin(); kt != jt->second.end(); ++kt)
							if (translation[*kt] == invalid)
								return match;
							else
								moved[jt->first].insert(translation[*kt]);
					}
					swap(match, moved);
				} while(true);
			}

			template<typename Subgraph, typename Graph, typename LatticeGraph> void lattice_constant_walker(
				  typename boost::graph_traits<Subgraph>::vertex_descriptor const & s
				, typename boost::graph_traits<Graph>::vertex_descriptor const & g
				, Subgraph const & S
				, Graph const & G
				, LatticeGraph const & LG
				, std::map<typename boost::graph_traits<Graph>::vertex_descriptor, std::size_t> & I
				, std::set<std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > > & matches
				, std::vector<std::vector<unsigned int> > const & translations
				, std::deque<std::pair<
					  typename boost::graph_traits<Subgraph>::vertex_descriptor
					, typename boost::graph_traits<Graph>::vertex_descriptor
				  > > stack
				, std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > match
				, std::set<typename boost::graph_traits<Subgraph>::vertex_descriptor> placed
				, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> visited
				, std::map<
					  typename boost::graph_traits<Subgraph>::vertex_descriptor
					, typename boost::graph_traits<Graph>::vertex_descriptor
				  > pinning
			) {
				if (out_degree(s, S) > out_degree(g, G))
					return;
				typename boost::graph_traits<Subgraph>::adjacency_iterator s_ai, s_ae;
				for (tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
					if (pinning.find(*s_ai) != pinning.end()) {
						typename boost::graph_traits<Graph>::edge_descriptor e;
						bool is_e;
						tie(e, is_e) = edge(g, pinning[*s_ai], G);
						// TODO: check colored graphs ...
						if (!is_e)
							return;
					}
				// TODO: check colored graphs ...
				visited.insert(g);
				if (match.find(I[s]) == match.end())
					match[I[s]] = std::set<typename boost::graph_traits<Graph>::vertex_descriptor>();
				match[I[s]].insert(g);
				pinning[s] = g;
				if (visited.size() < num_vertices(S)) {
					typename boost::graph_traits<Graph>::adjacency_iterator g_ai, g_ae;
					for (tie(s_ai, s_ae) = adjacent_vertices(s, S); s_ai != s_ae; ++s_ai)
						if (placed.find(*s_ai) == placed.end()) {
							placed.insert(*s_ai);
							stack.push_back(std::make_pair(*s_ai, g));
						}
					typename boost::graph_traits<Subgraph>::vertex_descriptor t = stack[0].first;
					tie(g_ai, g_ae) = adjacent_vertices(stack[0].second, G);
					stack.pop_front();
					for (; g_ai != g_ae; ++g_ai)
						if (visited.find(*g_ai) == visited.end())
							detail::lattice_constant_walker(t, *g_ai, S, G, LG, I, matches, translations, stack, match, placed, visited, pinning);
				} else {
					for (std::vector<std::vector<unsigned int> >::const_iterator it = translations.begin(); it != translations.end(); ++it)
						match = lattice_constant_translations(num_vertices(G), *it, match);
					if(matches.insert(match).second)
;//                        print_embedding_from_pinning(pinning,S,G,LG);
				}
			}
            template<typename Subgraph, typename Graph, typename Lattice, typename LatticeGraph>
			std::set<std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > > 
            lattice_embeddings(
                  Subgraph const & S
                , Graph const & G
                , Lattice const & L
                , LatticeGraph const & LG
                , std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & v
            ){
                typedef typename alps::graph_helper<Graph>::lattice_type lattice_type;
                typedef typename alps::lattice_traits<lattice_type>::unit_cell_type::graph_type unit_cell_graph_type;

                // TODO: implement that
                if (v.size() != 1)
                    ALPS_NGS_THROW_RUNTIME_ERROR("not Impl!")

                // Get the possible translation in the lattice
                std::vector<std::vector<unsigned int> > translations(detail::build_translation_table(G,L));
                
                // orbit index => vertices
                std::set<std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > > matches;
                typename partition_type<Graph>::type orbit = boost::get<2>(canonical_properties(S));
                std::map<typename boost::graph_traits<Subgraph>::vertex_descriptor, std::size_t> I;
                // Io = {(mi, j) : ni element of Vj
                detail::partition_indeces(I, orbit, S);

                for (typename partition_type<Subgraph>::type::const_iterator it = orbit.begin(); it != orbit.end(); ++it)
                    if (out_degree(it->front(), S) <= out_degree(v.front(), G)) {
                        std::set<typename boost::graph_traits<Subgraph>::vertex_descriptor> placed;
                        std::set<typename boost::graph_traits<Graph>::vertex_descriptor> visited;
                        std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > match;
                        std::deque<std::pair<
                              typename boost::graph_traits<Subgraph>::vertex_descriptor
                            , typename boost::graph_traits<Graph>::vertex_descriptor
                        > > stack;
                        std::map<
                              typename boost::graph_traits<Subgraph>::vertex_descriptor
                            , typename boost::graph_traits<Graph>::vertex_descriptor
                        > pinning;
						placed.insert(it->front());
                        detail::lattice_constant_walker(it->front(), v.front(), S, G, LG, I, matches, translations, stack, match, placed, visited, pinning);
                    }
                return matches;
            }
                        
		}

		// Input: Subgraph, Graph, vertices of G contained in mapping of S on G
		// Output: lattice_constant of S in G containing v
		template<typename Subgraph, typename Graph, typename Lattice, typename LatticeGraph> std::size_t lattice_constant(
			  Subgraph const & S
			, Graph const & G
			, Lattice const & L
			, LatticeGraph const & LG
			, std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> const & v
		) {
			std::set<std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > > matches = detail::lattice_embeddings(S, G, L, LG, v);
			return matches.size();
		}
        
       /*
       template <typename Subgraph, typename Graph, typename LatticeGraph> void print_embedding_from_match(
                 std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> > const& mapping
               , Subgraph const& S
               , Graph const& G
               , LatticeGraph const& LG
       ){
            typedef typename boost::graph_traits<LatticeGraph>::vertex_iterator vertex_iterator;
            typedef typename boost::graph_traits<LatticeGraph>::edge_iterator edge_iterator;

            std::cout<<"graph embedding {"<<std::endl;
            for(typename std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> >::const_iterator map_it= mapping.begin(); map_it != mapping.end(); ++map_it)
            {
                std::cout<<"// "<<map_it->first<<" -> ";
                for(typename std::set<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator it2 = map_it->second.begin(); it2 != map_it->second.end(); ++it2)
                {
                    std::cout<<*it2<<" ";
                }
                std::cout<<std::endl;
            }
            for(std::pair<vertex_iterator,vertex_iterator> vtcs = vertices(LG); vtcs.first != vtcs.second; ++vtcs.first)
            {
                std::vector<double> coord = boost::get(alps::coordinate_t(),LG,*vtcs.first);
                std::cout<<*vtcs.first;
                std::cout<<"[pos=\"";
                    typename std::vector<double>::iterator vit = coord.begin();
                    std::cout<<*vit++;
                    while(vit != coord.end())
                    {
                        std::cout<<","<<*vit;
                        ++vit;
                    }
                std::cout<<"!\"";

                    bool is_mapped = false;
                    for(typename std::map<unsigned, std::set<typename boost::graph_traits<Graph>::vertex_descriptor> >::const_iterator map_it= mapping.begin(); map_it != mapping.end(); ++map_it)
                    {
                        typename std::set<typename boost::graph_traits<Graph>::vertex_descriptor>::const_iterator lookup = std::find(map_it->second.begin(),map_it->second.end(),*vtcs.first);
                        if(lookup != map_it->second.end() )
                        {
                            is_mapped = true;
                            break;
                        }
                    }

                    if(is_mapped)
                        std::cout<<" color=\"red\"";
                    else
                        std::cout<<" color=\"gray\"";
                std::cout<<"];"<<std::endl;
            }
            
            std::pair<edge_iterator,edge_iterator> edge_range = edges(LG);
            while(edge_range.first != edge_range.second)
            {
                if(!boost::get(alps::boundary_crossing_t(),LG,*edge_range.first))
                {
                    typename boost::graph_traits<LatticeGraph>::vertex_descriptor s(source(*edge_range.first,LG)), t(target(*edge_range.first,LG));
                    std::cout<<s<<" -- "<<t<<";"<<std::endl;
                }
                ++edge_range.first;
            }
            std::cout<<"}"<<std::endl;
       }
       */
       
       
       template <typename Subgraph, typename Graph, typename LatticeGraph> void print_embedding_from_pinning(
				 std::map<
					  typename boost::graph_traits<Subgraph>::vertex_descriptor
					, typename boost::graph_traits<Graph>::vertex_descriptor
				  > const& pinning
               , Subgraph const& S
               , Graph const& G
               , LatticeGraph const& LG
       ){
            typedef typename boost::graph_traits<LatticeGraph>::vertex_iterator vertex_iterator;
            typedef typename boost::graph_traits<LatticeGraph>::edge_iterator edge_iterator;

		    typedef  std::map<
					  typename boost::graph_traits<Subgraph>::vertex_descriptor
					, typename boost::graph_traits<Graph>::vertex_descriptor
                    > pinning_type;
            typedef std::map<
                  typename boost::graph_traits<LatticeGraph>::vertex_descriptor
                , typename boost::graph_traits<Subgraph>::vertex_descriptor
                > pinning_inverse_type;
            pinning_inverse_type pinning_inverse;
            for(typename pinning_type::const_iterator it= pinning.begin(); it != pinning.end(); ++it)
            {
                pinning_inverse.insert(std::make_pair(it->second,it->first));
            }
            
            std::cout<<"graph embedding {"<<std::endl;
            std::cout<<"// "<<pinning.size()<<std::endl;
            for(std::pair<vertex_iterator,vertex_iterator> vtcs = vertices(LG); vtcs.first != vtcs.second; ++vtcs.first)
            {
                std::vector<double> coord = boost::get(alps::coordinate_t(),LG,*vtcs.first);
                std::cout<<*vtcs.first;
                std::cout<<"[pos=\"";
                    typename std::vector<double>::iterator vit = coord.begin();
                    std::cout<<*vit++;
                    while(vit != coord.end())
                    {
                        std::cout<<","<<*vit;
                        ++vit;
                    }
                std::cout<<"!\"";
                bool is_mapped = pinning_inverse.count(*vtcs.first);
                if(is_mapped)
                    std::cout<<" color=\"red\"";
                else
                    std::cout<<" color=\"gray\"";
                std::cout<<"];"<<std::endl;
            }
            
            std::pair<edge_iterator,edge_iterator> edge_range = edges(LG);
            while(edge_range.first != edge_range.second)
            {
                if(!boost::get(alps::boundary_crossing_t(),LG,*edge_range.first))
                {
                    typename boost::graph_traits<LatticeGraph>::vertex_descriptor s(source(*edge_range.first,LG)), t(target(*edge_range.first,LG));
                    std::cout<<s<<" -- "<<t;
                    bool is_mapped = false;
                    typename pinning_inverse_type::iterator s_lookup = pinning_inverse.find( s );
                    if(s_lookup != pinning_inverse.end())
                    {
                        typename pinning_inverse_type::iterator t_lookup = pinning_inverse.find( t );
                        if(t_lookup != pinning_inverse.end())
                        {
                            if(edge(s_lookup->second,t_lookup->second,S).second)
                                is_mapped = true;
                        }

                    }
                    if(is_mapped)
                        std::cout<<"[color=\"red\"]";
                    else
                        std::cout<<"[color=\"gray\"]";
                    std::cout<<";"<<std::endl;
                }
                ++edge_range.first;
            }
            std::cout<<"}"<<std::endl;
       }
	}
}
#endif
