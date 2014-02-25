/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Andreas Hehn <hehn@phys.ethz.ch>                   *
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
#include <alps/graph/utils.hpp>
#include <boost/static_assert.hpp>
#include <alps/graph/lattice_constant.hpp>
#include <alps/graph/is_embeddable.hpp>
#include <vector>

namespace alps {
namespace graph {

namespace detail {

    template <typename SubGraph, typename SuperGraph, typename CanoncialPropertiesPolicy>
    class subgraph_generator_impl_base : public CanoncialPropertiesPolicy
    {
      public:
        typedef SubGraph   subgraph_type;
        typedef SuperGraph supergraph_type;
        typedef std::pair<subgraph_type,typename canonical_properties_type<subgraph_type>::type> subgraph_properties_pair_type;
        typedef typename std::vector<subgraph_properties_pair_type>::iterator iterator;

        subgraph_generator_impl_base(supergraph_type const& supergraph, typename graph_traits<supergraph_type>::vertex_descriptor pin)
            : supergraph_(supergraph), graphs_(), pin_(pin), non_embeddable_graphs_(), labels_()
        {
        }

      protected:
        /*
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

        /*
          * Checks whether the subgraph g is embeddable in the supergraph.
          * For small graphs g the graph may be added to an internal list of non-embeddable graphs if it is not embeddable.
          * All subsequent calls to this function will first check agains this list. If any of these non-embeddable small graphs is embeddable in g, g is obviously not embeddable in the supergraph either.
          * \param g the graph to be checked
          * \param prop the canonical properties of g
          * \return true if g is embeddable, false if g is not embeddable
          */
        bool check_embedding(subgraph_type const& g, typename canonical_properties_type<subgraph_type>::type const& prop) {
            assert(prop == this->canonical_prop(g));
            typename graph_traits<subgraph_type>::edges_size_type const num_edges_g = num_edges(g);
            for(typename std::vector<std::pair<subgraph_type,typename partition_type<subgraph_type>::type> >::iterator it= non_embeddable_graphs_.begin(); it != non_embeddable_graphs_.end(); ++it)
            {
                // Graphs of the same size can't be embedded into each other unless they are the same
                if(num_edges(it->first) >= num_edges_g)
                    break;

                // If any of the non-embeddable graphs can be embedded
                // we can not embedd this graph either.
                if(this->try_to_embedd(it->first,g,it->second))
                    return false;
            }
            bool result = this->try_to_embedd_in_supergraph(g,supergraph_,pin_,boost::get<alps::graph::partition>(prop));
            if(!result) // && num_edges(g) < 9)
                non_embeddable_graphs_.push_back(std::make_pair(g,get<alps::graph::partition>(prop)));
            return result;
        }

        /// The supergraph for which the subgraphs are generated
        supergraph_type const supergraph_;
        /// The list of subgraphs that were found
        std::vector<subgraph_properties_pair_type> graphs_;
        /// A list of vertices of the supergraph which have to be found in an embeddable subgraph
        typename graph_traits<SuperGraph>::vertex_descriptor pin_;
        /// A list of small non-embeddable graphs for quick embedding checks of larger graphs
        std::vector<std::pair<subgraph_type, typename partition_type<subgraph_type>::type> > non_embeddable_graphs_;
        /// a list of canonical graph labels of graphs that were seen
        std::set<boost::tuple<std::size_t, typename graph_label<subgraph_type>::type> > labels_;
    };

    template <typename SubGraph, typename SuperGraph, typename EdgeColorSymmetryPolicy, bool ColoredVertices, bool ColoredEdges>
    class subgraph_generator_impl;

    template <typename SubGraph, typename SuperGraph, typename EdgeColorSymmetryPolicy>
    class subgraph_generator_impl<SubGraph, SuperGraph,EdgeColorSymmetryPolicy,false,false>
    : public subgraph_generator_impl_base<SubGraph, SuperGraph, EdgeColorSymmetryPolicy>
    {
      public:
        typedef subgraph_generator_impl_base<SubGraph, SuperGraph, EdgeColorSymmetryPolicy>  base_type;

        typedef typename base_type::subgraph_type                   subgraph_type;
        typedef typename base_type::supergraph_type                 supergraph_type;
        typedef typename base_type::subgraph_properties_pair_type   subgraph_properties_pair_type;
        typedef typename base_type::iterator                        iterator;

        subgraph_generator_impl(supergraph_type const& g, typename graph_traits<supergraph_type>::vertex_descriptor pin)
            : base_type(g,pin)
        {
            analyze_supergraph(g);
        }

        void analyze_supergraph(SuperGraph const& g)
        {
            // Analyse vertices
            max_degree_ = 0;
            typename graph_traits<supergraph_type>::vertex_iterator v_it, v_end;
            for(boost::tie(v_it,v_end) = vertices(g); v_it != v_end; ++v_it)
                max_degree_ = (std::max)(max_degree_,degree(*v_it, g));
        }

        void initialize()
        {
            this->graphs_.clear();
            // Initialize
            subgraph_type sg;
            add_vertex(sg);
            this->graphs_.push_back(std::make_pair(sg,this->canonical_prop(sg)));
        }

        void grow_at(std::vector<subgraph_properties_pair_type>& result, iterator const it, typename graph_traits<subgraph_type>::vertex_descriptor v, typename partition_type<subgraph_type>::type::const_iterator const p_it)
        {
            // Skip the vertex if it can't have more edges
            if( degree(v, it->first) >= max_degree_ )
                return;
            grow_new_vertex(result,it,v);
            grow_new_edges(result,it,v,p_it);
        }

        void grow_new_vertex(std::vector<subgraph_properties_pair_type>& result, iterator const it, typename graph_traits<subgraph_type>::vertex_descriptor v)
        {
            // Create a new graph by adding a edge to a new vertex
            typedef typename canonical_properties_type<subgraph_type>::type canonical_properties_type;

            subgraph_type new_graph(it->first);
            add_edge( v, add_vertex(new_graph), new_graph);
            canonical_properties_type const new_graph_prop = this->canonical_prop(new_graph);
            if( this->is_unknown(new_graph_prop) && this->check_embedding(new_graph,new_graph_prop) )
                result.push_back(std::make_pair(new_graph,new_graph_prop));
        }

        void grow_new_edges(std::vector<subgraph_properties_pair_type>& result, iterator const it, typename graph_traits<subgraph_type>::vertex_descriptor v,  typename partition_type<subgraph_type>::type::const_iterator const p_it)
        {
            // Create a new graph by drawing edges between existing vertices
            typedef typename graph_traits<subgraph_type>::vertex_descriptor vertex_descriptor;
            typedef typename partition_type<subgraph_type>::type            partition_type;
            typedef typename canonical_properties_type<subgraph_type>::type canonical_properties_type;

            for(typename partition_type::const_iterator p_it2 = get<alps::graph::partition>(it->second).begin(); p_it2 != p_it; ++p_it2) {
                vertex_descriptor v2 = *(p_it2->begin());
                // Skip the vertex if it can't have more edges
                if(degree(v2,it->first) >= max_degree_)
                    continue;

                // Add only edges between vertices which are not connected
                // by an edge yet (prevent "double edges")
                if(!edge(v, v2, it->first).second) {
                    subgraph_type new_graph2(it->first);
                    add_edge( v2, v, new_graph2);
                    canonical_properties_type const new_graph2_prop = this->canonical_prop(new_graph2);
                    if( this->is_unknown(new_graph2_prop) && this->check_embedding(new_graph2,new_graph2_prop) )
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
                    canonical_properties_type const new_graph2_prop = this->canonical_prop(new_graph2);
                    if( this->is_unknown(new_graph2_prop) && this->check_embedding(new_graph2,new_graph2_prop) )
                        result.push_back(std::make_pair(new_graph2,new_graph2_prop));
                }
            }
        }

      private:
        typename graph_traits<supergraph_type>::degree_size_type max_degree_;
    };

    template <typename SubGraph, typename SuperGraph, typename EdgeColorSymmetryPolicy>
    class subgraph_generator_impl<SubGraph, SuperGraph, EdgeColorSymmetryPolicy, false, true>
    : public subgraph_generator_impl_base<SubGraph, SuperGraph, EdgeColorSymmetryPolicy>
    {
      public:
        typedef subgraph_generator_impl_base<SubGraph, SuperGraph, EdgeColorSymmetryPolicy>  base_type;

        typedef typename base_type::subgraph_type                   subgraph_type;
        typedef typename base_type::supergraph_type                 supergraph_type;
        typedef typename base_type::subgraph_properties_pair_type   subgraph_properties_pair_type;
        typedef typename base_type::iterator                        iterator;

        typedef std::vector<typename has_property<alps::edge_type_t,supergraph_type>::edge_property_type> edge_color_list_type;
        typedef boost::tuple<unsigned int, edge_color_list_type> supergraph_info_type;

        subgraph_generator_impl(supergraph_type const& g, typename graph_traits<supergraph_type>::vertex_descriptor pin)
            : base_type(g,pin)
        {
            analyze_supergraph(g);
        }

        void analyze_supergraph(supergraph_type const& g)
        {
            using std::sort;
            using std::unique;

            // Analyse vertices
            max_degree_ = 0;
            typename graph_traits<supergraph_type>::vertex_iterator v_it, v_end;
            for(boost::tie(v_it,v_end) = vertices(g); v_it != v_end; ++v_it)
                max_degree_ = (std::max)(max_degree_,degree(*v_it, g));

            // Analyse edges
            edge_colors_ = get_color_list(alps::edge_type_t(),g);
        }

        void initialize()
        {
            this->graphs_.clear();
            // Initialize
            subgraph_type sg;
            add_vertex(sg);
            this->graphs_.push_back(std::make_pair(sg,this->canonical_prop(sg)));
        }

        void grow_at(std::vector<subgraph_properties_pair_type>& result, iterator const it, typename graph_traits<subgraph_type>::vertex_descriptor v,  typename partition_type<subgraph_type>::type::const_iterator const p_it)
        {
            // Skip the vertex if it can't have more edges
            if( degree(v, it->first) >= max_degree_ )
                return;
            grow_new_vertex(result,it,v);
            grow_new_edges(result,it,v,p_it);

        }

        void grow_new_vertex(std::vector<subgraph_properties_pair_type>& result, iterator const it, typename graph_traits<subgraph_type>::vertex_descriptor v)
        {
            // Create a new graph by adding a edge to a new vertex
            typedef typename canonical_properties_type<subgraph_type>::type canonical_properties_type;

            subgraph_type new_graph(it->first);
            typename graph_traits<subgraph_type>::edge_descriptor new_edge = add_edge( v, add_vertex(new_graph), new_graph).first;

            for(typename edge_color_list_type::iterator ecl_it = edge_colors_.begin(); ecl_it != edge_colors_.end(); ++ecl_it)
            {
                put( alps::edge_type_t(), new_graph, new_edge, *ecl_it);
                canonical_properties_type const new_graph_prop = this->canonical_prop(new_graph);
                if( this->is_unknown(new_graph_prop) && this->check_embedding(new_graph,new_graph_prop) )
                    result.push_back(std::make_pair(new_graph,new_graph_prop));
            }
        }

        void grow_new_edges(std::vector<subgraph_properties_pair_type>& result, iterator const it, typename graph_traits<subgraph_type>::vertex_descriptor v,  typename partition_type<subgraph_type>::type::const_iterator const p_it)
        {
            // Create a new graph by drawing edges between existing vertices
            typedef typename graph_traits<subgraph_type>::vertex_descriptor vertex_descriptor;
            typedef typename canonical_properties_type<subgraph_type>::type canonical_properties_type;
            typedef typename partition_type<subgraph_type>::type            partition_type;

            for(typename partition_type::const_iterator p_it2 = get<alps::graph::partition>(it->second).begin(); p_it2 != p_it; ++p_it2) {
                vertex_descriptor v2 = *(p_it2->begin());
                // Skip the vertex if it can't have more edges
                if(degree(v2,it->first) >= max_degree_)
                    continue;

                // Add only edges between vertices which are not connected
                // by an edge yet (prevent "double edges")
                if(!edge(v, v2, it->first).second) {
                    subgraph_type new_graph2(it->first);
                    typename graph_traits<subgraph_type>::edge_descriptor new_edge = add_edge( v2, v, new_graph2).first;
                    for(typename edge_color_list_type::iterator ecl_it = edge_colors_.begin(); ecl_it != edge_colors_.end(); ++ecl_it)
                    {
                        put( alps::edge_type_t(), new_graph2, new_edge, *ecl_it);
                        canonical_properties_type const new_graph2_prop = this->canonical_prop(new_graph2);
                        if( this->is_unknown(new_graph2_prop) && this->check_embedding(new_graph2,new_graph2_prop) )
                            result.push_back(std::make_pair(new_graph2,new_graph2_prop));
                    }
                }
            }

            // Create a new edge between vertices of the same orbit
            if((p_it->begin()+1) != p_it->end()) {
                vertex_descriptor v2 = *(p_it->begin()+1);

                // Add only edges between vertices which are not connected
                // by an edge yet (prevent "double edges")
                if(!edge(v, v2, it->first).second) {
                    subgraph_type new_graph2(it->first);
                    typename graph_traits<subgraph_type>::edge_descriptor new_edge = add_edge( v2, v, new_graph2).first;
                    for(typename edge_color_list_type::iterator ecl_it = edge_colors_.begin(); ecl_it != edge_colors_.end(); ++ecl_it)
                    {
                        put( alps::edge_type_t(), new_graph2, new_edge, *ecl_it);
                        canonical_properties_type const new_graph2_prop = this->canonical_prop(new_graph2);
                        if( this->is_unknown(new_graph2_prop) && this->check_embedding(new_graph2,new_graph2_prop) )
                            result.push_back(std::make_pair(new_graph2,new_graph2_prop));
                    }
                }
            }
        }

        private:
            typename graph_traits<supergraph_type>::degree_size_type max_degree_;
            edge_color_list_type edge_colors_;
    };
} // end namespace detail

namespace policies {

    /**
      * \brief A policy class for the subgraph_generator class
      *
      * This is the default policy to create canonical labels and check
      * embeddings for sub graphs within the subgraph_generator
      * \tparam SubGraph { The type of the subgraphs to be generated.
      * Has to fulfill the concepts required by canonical_properties()
      * and is_embeddable(). }
      */
    template <typename SubGraph>
    struct no_color_symmetry
    {
        inline typename canonical_properties_type<SubGraph>::type canonical_prop(SubGraph const& g)
        {
            return canonical_properties(g);
        }

        template <typename Graph>
        inline bool try_to_embedd(SubGraph const& sg, Graph const& g, typename partition_type<SubGraph>::type const& sg_prop)
        {
            using alps::graph::is_embeddable;
            return is_embeddable(sg,g,sg_prop);
        }
        template <typename Graph>
        inline bool try_to_embedd_in_supergraph(SubGraph const& sg, Graph const& g, typename boost::graph_traits<Graph>::vertex_descriptor pin, typename partition_type<SubGraph>::type const& sg_prop)
        {
            using alps::graph::is_embeddable;
            return is_embeddable(sg,g,pin,sg_prop);
        }
    };

    /**
      * \brief A policy class for the subgraph_generator class
      *
      * This is a special policy to create canonical labels and check embeddings
      * for sub graphs within the subgraph_generator considering permutation
      * symmetries of the edge colors.
      *
      * WARNING: While embedding to the given super graph the symmetry is
      * ignored. It is assumed all sub graph within a symmetry group can be
      * embedded in the super graph in a similar way.
      *
      * After construction of the subgraph_generator object, the member
      * set_color_partition(...) has to be called to pass the symmetry
      * information to the object.
      * \tparam SubGraph { The type of the subgraphs to be generated.
      * Has to fulfill the concepts required by canonical_properties()
      * and is_embeddable(). }
      */
    template <typename SubGraph>
    struct edge_color_symmetries
    {
        inline typename canonical_properties_type<SubGraph>::type canonical_prop(SubGraph const& g)
        {
            assert(!c_.empty());
            return canonical_properties(g,c_);
        }

        template <typename Graph>
        inline bool try_to_embedd(SubGraph const& sg, Graph const& g, typename partition_type<SubGraph>::type const& sg_prop)
        {
            using alps::graph::is_embeddable;
            return is_embeddable(sg,g,sg_prop,c_);
        }

        template <typename Graph>
        inline bool try_to_embedd_in_supergraph(SubGraph const& sg, Graph const& g, typename boost::graph_traits<Graph>::vertex_descriptor pin, typename partition_type<SubGraph>::type const& sg_prop)
        {
            using alps::graph::is_embeddable;
            return is_embeddable(sg,g,pin,get<alps::graph::partition>(canonical_properties(sg)));
        }

        /**
          * Sets the edge color permutation symmetry.
          * \param color_partitions a map assigning each edge color a group number. The graph is symmetric under permutations of colors having the same group number.
          */
        void set_color_partition(typename color_partition<SubGraph>::type const& color_partitions)
        {
            c_ = color_partitions;
        }
      private:
        typename color_partition<SubGraph>::type c_;
    };
} // end namespace policies

    /**
      * \brief the subgraph_generator class
      *
      * A class for generating subgraphs of an (super-)graph, e.g. a lattice.
      * \tparam SubGraph the type of the subgraphs to be generated. Has to fulfill boost::MutableGraph, boost::IncidenceGraph concepts and the concepts required by canonical_properties() and embedding().
      * \tparam SuperGraph the type of the (super-)graph for which the subgraphs are generated. Has to fulfill boost::IncidenceGraph, boost::EdgeListGraph, boost::VertexListGraph concepts and the concepts required by embedding().
      * \tparam EdgeColorSymmetryPolicy the policy how to create canonical_properties for the generated subgraphs. Must offer at least the functions of policies::canonical_properties_simple_policy policy class.
      */
template <typename SubGraph, typename SuperGraph, typename EdgeColorSymmetryPolicy = policies::no_color_symmetry<SubGraph> >
class subgraph_generator
: public detail::subgraph_generator_impl<SubGraph, SuperGraph, EdgeColorSymmetryPolicy, has_property<alps::vertex_type_t,SubGraph>::vertex_property, has_property<alps::edge_type_t,SubGraph>::edge_property>
{
  public:
    typedef detail::subgraph_generator_impl<SubGraph, SuperGraph, EdgeColorSymmetryPolicy, has_property<alps::vertex_type_t,SubGraph>::vertex_property, has_property<alps::edge_type_t,SubGraph>::edge_property> base_type;

    typedef typename base_type::subgraph_type                   subgraph_type;
    typedef typename base_type::supergraph_type                 supergraph_type;
    typedef typename base_type::subgraph_properties_pair_type   subgraph_properties_pair_type;
    typedef typename base_type::iterator                        iterator;

    /**
      * Constructor
      * \param supergraph the supergraph for which the graphs will be generated.
      * \param pins is a list of vertices of the supergraph. A subgraph is only embeddable if the subgraph can be embedded in such a way that all those vertices have corresponding vertices in the subgraph.
     */
    subgraph_generator(supergraph_type const& supergraph, typename graph_traits<supergraph_type>::vertex_descriptor pin)
        : base_type(supergraph,pin)
    {
        // We assume undirected graphs
        BOOST_STATIC_ASSERT(( boost::is_same<typename graph_traits<subgraph_type>::directed_category, boost::undirected_tag>::value ));
        BOOST_STATIC_ASSERT(( boost::is_same<typename graph_traits<supergraph_type>::directed_category, boost::undirected_tag>::value ));
    }

    /**
      * Generates all subgraphs with up to n edges
      * \param n the upper limit of edges of the subgraphs to be generated
      * \return a std::pair of iterators pointing to the beginning and the end of the list of subgraphs
      */
    std::pair<iterator,iterator> generate_up_to_n_edges(unsigned int n)
    {
        if(this->graphs_.empty())
            this->initialize();
        iterator cur_end(this->graphs_.end());
        iterator last_end(this->graphs_.begin());

        // While the last graph has not the desired number of edges
        while(num_edges(this->graphs_.back().first) < n)
        {
            // Generate all graphs with N edges from the graphs with N-1 edges
            std::vector<subgraph_properties_pair_type> new_graphs(generate_graphs_with_additional_edge(last_end, cur_end));

            // Abort if no new graphs were found
            if(new_graphs.size() == 0)
                break;

            std::cout<<num_edges(new_graphs.back().first)<<":"<< new_graphs.size()<<std::endl;

            // Reserve the space required to append the generated graphs to our graphs,
            // so our last_end iterator doesn't get invalidated
            this->graphs_.reserve(this->graphs_.size()+new_graphs.size());
            last_end = this->graphs_.end();

            // Actually append the generated graphs to our vector
            this->graphs_.insert( this->graphs_.end(), new_graphs.begin(), new_graphs.end() );
            cur_end = this->graphs_.end();
        }
        return std::make_pair(this->graphs_.begin(),this->graphs_.end());
    }


    /**
      * Generates all subgraphs with exactly n edges
      * \param n the number of edges of the subgraphs to be generated
      * \return a std::pair of iterators pointing to the beginning and the end of the list of subgraphs
      */
    std::pair<iterator,iterator> generate_exactly_n_edges(unsigned int n) {
        if(this->graphs_.empty())
            this->initialize();
        // While the last graph has not the desired number of edges
        // and we found new graphs in the last iteration
        while( (num_edges(this->graphs_.back().first) < n) && (this->graphs_.size() == 0) )
            this->graphs_ = generate_graphs_with_additional_edge(this->graphs_.begin(), this->graphs_.end());

        return std::make_pair(this->graphs_.begin(), this->graphs_.end() );
    }

 private:
    /**
      * Generates a new graphs by adding one new edge (and maybe a vertex)
      * to the graphs given by the range of iterators.
      * \param it the beginning of the range
      * \param end the end of the range
      * \return a std::vector of the new graphs with the additional edge
      */
      std::vector<subgraph_properties_pair_type> generate_graphs_with_additional_edge(iterator it, iterator const end)
      {
          using boost::get;
          typedef typename graph_traits<subgraph_type>::vertex_descriptor vertex_descriptor;
          typedef typename partition_type<subgraph_type>::type            partition_type;

          std::vector<subgraph_properties_pair_type> result;
          while( it != end )
          {
              // Get the partition and iterate over one vertex per orbit (and not every single vertex)
              partition_type const& graph_partition = get<alps::graph::partition>(it->second);
              for (typename partition_type::const_iterator p_it = graph_partition.begin(); p_it != graph_partition.end(); ++p_it) {
                  vertex_descriptor v = *(p_it->begin());
                  this->grow_at(result, it, v, p_it);
              }
              ++it;
          }
          return result;
      }
};

template <typename SubGraph, typename SuperGraph>
std::vector<std::pair<SubGraph, typename canonical_properties_type<SubGraph>::type> > generate_subgraphs(SubGraph const&, SuperGraph const& supergraph, typename boost::graph_traits<SuperGraph>::vertex_descriptor pin, unsigned int n)
{
    typedef std::vector<std::pair<SubGraph, typename canonical_properties_type<SubGraph>::type> > subgraph_list_type;
    typedef typename subgraph_list_type::iterator iterator;
    subgraph_generator<SubGraph, SuperGraph> sg(supergraph,pin);
    iterator it, end;
    boost::tie(it,end) = sg.generate_up_to_n_edges(n);
    return subgraph_list_type(it,end);
}

template <typename SubGraph, typename SuperGraph>
std::vector<std::pair<SubGraph, typename canonical_properties_type<SubGraph>::type> > generate_subgraphs(SubGraph const&, SuperGraph const& supergraph, typename boost::graph_traits<SuperGraph>::vertex_descriptor pin, unsigned int n, typename color_partition<SubGraph>::type const& color_partitions)
{
    typedef std::vector<std::pair<SubGraph, typename canonical_properties_type<SubGraph>::type> > subgraph_list_type;
    typedef typename subgraph_list_type::iterator iterator;
    subgraph_generator<SubGraph, SuperGraph, policies::edge_color_symmetries<SubGraph> > sg(supergraph,pin);
    sg.set_color_partition(color_partitions);
    iterator it, end;
    boost::tie(it,end) = sg.generate_up_to_n_edges(n);
    return subgraph_list_type(it,end);
}


} // namespace graph
} // namespace alps

#endif //ALPS_GRAPH_SUBGRAPH_GENERATOR_HPP
