/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Andreas Hehn <hehn@phys.ethz.ch>                   *
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

#ifndef ALPS_GRAPH_SUBGRAPH_GENERATOR_HPP
#define ALPS_GRAPH_SUBGRAPH_GENERATOR_HPP
#include <alps/lattice.h>
#include <alps/graph/canonical_properties.hpp>
#include <boost/static_assert.hpp>
#include <alps/graph/lattice_constant.hpp>
#include <vector>

namespace alps {
namespace graph {

    /**
      * \brief the subgraph_generator class
      *
      * A class for generating subgraphs of an (super-)graph, e.g. a lattice.
      * \tparam SubGraph the type of the subgraphs to be generated. Has to fulfill boost::MutableGraph, boost::IncidenceGraph concepts and the concepts required by canonical_properties() and embedding().
      * \tparam SuperGraph the type of the (super-)graph for which the subgraphs are generated. Has to fulfill boost::IncidenceGraph, boost::EdgeListGraph, boost::VertexListGraph concepts and the concepts required by embedding().
      */
template <typename SubGraph, typename SuperGraph>
class subgraph_generator {
  public:
    typedef SubGraph subgraph_type;
    typedef std::pair<subgraph_type,typename canonical_properties_type<subgraph_type>::type> subgraph_properties_pair_type;
    typedef SuperGraph supergraph_type;
    typedef typename std::vector<subgraph_properties_pair_type>::iterator iterator;

    /**
      * Constructor
      * Initializes all members and makes a call to analyse_supergraph()
      * \param supergraph the supergraph for which the graphs will be generated.
      * \param pins is a list of vertices of the supergraph. A subgraph is only embeddable if the subgraph can be embedded in such a way that all those vertices have corresponding vertices in the subgraph.
     */
    subgraph_generator(supergraph_type const& supergraph, typename graph_traits<supergraph_type>::vertex_descriptor pin)
        : supergraph_(supergraph), graphs_(), max_degree_(0), non_embeddable_graphs_(), labels_(), pin_(pin) {
        // We assume undirected graphs
        BOOST_STATIC_ASSERT(( boost::is_same<typename graph_traits<subgraph_type>::directed_category, boost::undirected_tag>::value ));
        BOOST_STATIC_ASSERT(( boost::is_same<typename graph_traits<supergraph_type>::directed_category, boost::undirected_tag>::value ));

        //TODO get properties of the vertex from the lattice
        subgraph_type g;
        add_vertex(g);
        graphs_.push_back(std::make_pair(g,canonical_properties(g)));

        analyse_supergraph();
    }

    /**
      * Generates all subgraphs with up to n edges
      * \param n the upper limit of edges of the subgraphs to be generated
      * \return a std::pair of iterators pointing to the beginning and the end of the list of subgraphs
      */
    std::pair<iterator,iterator> generate_up_to_n_edges(unsigned int n) {
        iterator cur_end(graphs_.end());
        iterator last_end(graphs_.begin());

        // While the last graph has not the desired number of edges
        while(num_edges(graphs_.back().first) < n)
        {
            // Generate all graphs with N edges from the graphs with N-1 edges
            std::vector<subgraph_properties_pair_type> new_graphs(generate_graphs_with_additional_edge(last_end, cur_end));

            // Abort if no new graphs were found
            if(new_graphs.size() == 0)
                break;

            std::cout<<num_edges(new_graphs.back().first)<<":"<< new_graphs.size()<<std::endl;
            
            // Reserve the space required to append the generated graphs to our graphs,
            // so our last_end iterator doesn't get invalidated
            graphs_.reserve(graphs_.size()+new_graphs.size());
            last_end = graphs_.end();
            
            // Actually append the generated graphs to our vector
            graphs_.insert( graphs_.end(), new_graphs.begin(), new_graphs.end() );
            cur_end = graphs_.end();
        }
        return std::make_pair(graphs_.begin(),graphs_.end());
    }


    /**
      * Generates all subgraphs with exactly n edges
      * \param n the number of edges of the subgraphs to be generated
      * \return a std::pair of iterators pointing to the beginning and the end of the list of subgraphs
      */
    std::pair<iterator,iterator> generate_exactly_n_edges(unsigned int n) {
        assert( graphs_.size() >= 0 );
        // While the last graph has not the desired number of edges
        // and we found new graphs in the last iteration
        while( (num_edges(graphs_.back().first) < n) && (graphs_.size() == 0) )
            graphs_ = generate_graphs_with_additional_edge(graphs_.begin(), graphs_.end());

        return std::make_pair(graphs_.begin(), graphs_.end() );
    }

 private:
    /**
      * Generates a new graphs by adding one new edge (and maybe a vertex)
      * to the graphs given by the range of iterators.
      * \param it the beginning of the range
      * \param end the end of the range
      * \return a std::vector of the new graphs with the additional edge
      */
    std::vector<subgraph_properties_pair_type> generate_graphs_with_additional_edge(iterator it, iterator const end) {
        using boost::get;
        typedef typename graph_traits<subgraph_type>::vertex_descriptor vertex_descriptor;
        typedef typename canonical_properties_type<subgraph_type>::type canonical_properties_type;
        typedef typename partition_type<subgraph_type>::type partition_type;

        std::vector<subgraph_properties_pair_type> result;
        while( it != end ) {
            // Get the partition and iterate over one vertex per orbit (and not every single vertex)
            partition_type const graph_partition = get<alps::graph::partition>(it->second);
            for (typename partition_type::const_iterator p_it = graph_partition.begin(); p_it != graph_partition.end(); ++p_it) {
                vertex_descriptor v = *(p_it->begin());
                // Skip the vertex if it can't have more edges
                if(degree(v,it->first) >= max_degree_)
                    continue;
                
                // Create a new graph by adding a edge to a new vertex
                {
                    subgraph_type new_graph(it->first);
                    add_edge( v, add_vertex(new_graph), new_graph);
                    canonical_properties_type const new_graph_prop = canonical_properties(new_graph);
                    if( is_unknown(new_graph_prop) && is_embeddable(new_graph,new_graph_prop) )
                        result.push_back(std::make_pair(new_graph,new_graph_prop));
                }
            
                // Create a new graph by drawing edges between existing vertices
                for(typename partition_type::const_iterator p_it2 = graph_partition.begin(); p_it2 != p_it; ++p_it2) {
                    vertex_descriptor v2 = *(p_it2->begin());
                    // Skip the vertex if it can't have more edges
                    if(degree(v2,it->first) >= max_degree_)
                        continue;

                    // Add only edges between vertices which are not connected
                    // by an edge yet (prevent "double edges")
                    if(!edge(v, v2, it->first).second) {
                        subgraph_type new_graph2(it->first);
                        add_edge( v2, v, new_graph2);
                        canonical_properties_type const new_graph2_prop = canonical_properties(new_graph2);
                        if( is_unknown(new_graph2_prop) && is_embeddable(new_graph2,new_graph2_prop) )
                            result.push_back(std::make_pair(new_graph2,new_graph2_prop));
                    }
                }

                // Create a new edge between vertices of the same orbit
                if((p_it->begin()+1) != p_it->end()) {
                    vertex_descriptor v2 = *(p_it->begin()+1);

                    // Add only edges between vertices which are not connected
                    // by an edge yet (prevent "double edges")
                    if(!edge(v, v2, it->first).second) {
                        subgraph_type new_graph2(it->first);
                        add_edge( v2, v, new_graph2);
                        canonical_properties_type const new_graph2_prop = canonical_properties(new_graph2);
                        if( is_unknown(new_graph2_prop) && is_embeddable(new_graph2,new_graph2_prop) )
                            result.push_back(std::make_pair(new_graph2,new_graph2_prop));
                    }
                }
            }
            ++it;
        }
        return result;
    }

    /**
      * Checks whether the canonical label inside the canonical_properties p of some graph is already known.
      * This function is used to drop graphs for which an isomorphic graph was already found.
      * This function modifies the state of the object as it adds the canonical label to the known labels list.
      * \param p the canonical_properties of the graph to be checked
      * \return true is the graph(-label) was unknown, false if the label has been seen before.
      */
    bool is_unknown(typename canonical_properties_type<subgraph_type>::type const& p) {
        typename graph_label<subgraph_type>::type const& label(boost::get<alps::graph::label>(p));
        // Try to insert the label and return true if it wasn't there yet
        return labels_.insert(make_tuple(boost::get<0>(label).size(),label)).second;
    }
    
    /**
      * Checks whether the subgraph g is embeddable in the supergraph.
      * For small graphs g the graph may be added to an internal list of non-embeddable graphs if it is not embeddable.
      * All subsequent calls to this function will first check agains this list. If any of these non-embeddable small graphs is embeddable in g, g is obviously not embeddable in the supergraph either.
      * \param g the graph to be checked
      * \param prop the canonical properties of g
      * \return true if g is embeddable, false if g is not embeddable
      */
    bool is_embeddable(subgraph_type const& g, typename canonical_properties_type<subgraph_type>::type const& prop) {
        assert(prop == alps::graph::canonical_properties(g));
        bool result = true;
        for(typename std::vector<subgraph_properties_pair_type>::iterator it= non_embeddable_graphs_.begin(); it != non_embeddable_graphs_.end(); ++it)
        {
            // If any of the non-embeddable graphs can be embedded
            // we can not embedd this graph either.
            if(alps::graph::is_embeddable(it->first,g,get<alps::graph::partition>(it->second))) {
                result = false;
                break;
            }
        }
        result = result && alps::graph::is_embeddable(g,supergraph_,pin_,boost::get<alps::graph::partition>(prop));
        if(!result && num_edges(g) < 9)
            non_embeddable_graphs_.push_back(std::make_pair(g,prop));
        return result;
    }
    
    /**
      * Makes an initial analysis of the supergraph to generate criteria which all subgraphs have to fullfil.
      * E.g. the maximal degree of a single site in the graph.
      */
    void analyse_supergraph() {
        max_degree_ = 0;
        typename graph_traits<supergraph_type>::vertex_iterator v_it, v_end;
        for(boost::tie(v_it,v_end) = vertices(supergraph_); v_it != v_end; ++v_it)
            max_degree_ = std::max(max_degree_,degree(*v_it, supergraph_));
    }
    
    //
    // Data members
    //

    /// The supergraph for which the subgraphs are generated
    supergraph_type const supergraph_;
    /// The list of subgraphs that were found
    std::vector<subgraph_properties_pair_type> graphs_;
    /// properties of the supergraph for simple checks
    typename graph_traits<supergraph_type>::degree_size_type max_degree_;
    /// A list of small non-embeddable graphs for quick embedding checks of larger graphs
    std::vector<subgraph_properties_pair_type> non_embeddable_graphs_;
    
    /// a list of canonical graph labels of graphs that were seen
    std::set<boost::tuple<std::size_t, typename graph_label<subgraph_type>::type> > labels_;
    /// A list of vertices of the supergraph which have to be found in an embeddable subgraph
    typename graph_traits<SuperGraph>::vertex_descriptor pin_;
};

} // namespace graph
} // namespace alps

#endif //ALPS_GRAPH_SUBGRAPH_GENERATOR_HPP
