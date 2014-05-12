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

#include <alps/graph/subgraph_generator.hpp>
#include <alps/graph/utils.hpp>
#include <boost/graph/adjacency_list.hpp>

static unsigned int const test_graph_size = 6;

template <typename Graph>
unsigned int generate_graphs_with_edge_color_symmetry(unsigned int order)
{
    std::ifstream in("../../lib/xml/lattices.xml");
    alps::Parameters parm;
    parm["LATTICE"] = "anisotropic triangular lattice";
    parm["L"]       = 2*order+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef alps::coordinate_graph_type lattice_graph_type;
    lattice_graph_type& lattice_graph = alps_lattice.graph();

    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type,alps::graph::policies::edge_color_symmetries<Graph> > graph_gen_type;
    graph_gen_type graph_gen(lattice_graph, 2*order*order);

    typename alps::graph::color_partition<Graph>::type color_sym_group;
    color_sym_group[0] = 0;
    color_sym_group[1] = 0;
    color_sym_group[2] = 0;
    graph_gen.set_color_partition(color_sym_group);

    typename graph_gen_type::iterator it,end;

    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order);
    return std::distance(it,end);
}

template <typename Graph>
unsigned int generate_graphs_without_symmetry(unsigned int order)
{
    using alps::graph::canonical_properties;
    std::ifstream in("../../lib/xml/lattices.xml");
    alps::Parameters parm;
    parm["LATTICE"] = "anisotropic triangular lattice";
    parm["L"]       = 2*order+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef alps::coordinate_graph_type lattice_graph_type;
    lattice_graph_type& lattice_graph = alps_lattice.graph();

    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type> graph_gen_type;
    typename graph_gen_type::iterator it,end;
    graph_gen_type graph_gen(lattice_graph, 2*order*order);
    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order);


    // Identify symmetric graphs
    typename alps::graph::color_partition<Graph>::type color_sym_group;
    color_sym_group[0] = 0;
    color_sym_group[1] = 0;
    color_sym_group[2] = 0;

    std::set<std::pair<std::size_t,typename alps::graph::graph_label<Graph>::type> > non_symmetric_graphs;
    for( ;it != end; ++it)
        non_symmetric_graphs.insert(std::make_pair(num_vertices(it->first),get<alps::graph::label>(canonical_properties(it->first,color_sym_group))));
    return non_symmetric_graphs.size();
}

template <typename Graph>
std::map<typename alps::graph::graph_label<Graph>::type,Graph> generate_graphs_with_edge_color_symmetry_g(unsigned int order)
{
    std::ifstream in("../../lib/xml/lattices.xml");
    alps::Parameters parm;
    parm["LATTICE"] = "anisotropic triangular lattice";
    parm["L"]       = 2*order+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef alps::coordinate_graph_type lattice_graph_type;
    lattice_graph_type& lattice_graph = alps_lattice.graph();

    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type,alps::graph::policies::edge_color_symmetries<Graph> > graph_gen_type;
    graph_gen_type graph_gen(lattice_graph, 2*order*order);

    typename alps::graph::color_partition<Graph>::type color_sym_group;
    color_sym_group[0] = 0;
    color_sym_group[1] = 0;
    color_sym_group[2] = 0;
    graph_gen.set_color_partition(color_sym_group);

    typename graph_gen_type::iterator it,end;

    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order);
//    return std::distance(it,end);
    std::map<typename alps::graph::graph_label<Graph>::type,Graph> ret;
    for(; it != end; ++it)
        ret.insert(std::make_pair(get<alps::graph::label>(it->second),it->first));
    return ret;
}
template <typename Graph>
std::map<typename alps::graph::graph_label<Graph>::type,Graph> generate_graphs_without_symmetry_g(unsigned int order)
{
    using alps::graph::canonical_properties;
    std::ifstream in("../../lib/xml/lattices.xml");
    alps::Parameters parm;
    parm["LATTICE"] = "anisotropic triangular lattice";
    parm["L"]       = 2*order+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef alps::coordinate_graph_type lattice_graph_type;
    lattice_graph_type& lattice_graph = alps_lattice.graph();

    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type> graph_gen_type;
    typename graph_gen_type::iterator it,end;
    graph_gen_type graph_gen(lattice_graph, 2*order*order);
    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order);


    // Identify symmetric graphs
    typename alps::graph::color_partition<Graph>::type color_sym_group;
    color_sym_group[0] = 0;
    color_sym_group[1] = 0;
    color_sym_group[2] = 0;

    std::map<typename alps::graph::graph_label<Graph>::type,Graph> non_symmetric_graphs;
    for( ;it != end; ++it)
        non_symmetric_graphs.insert(std::make_pair(get<alps::graph::label>(canonical_properties(it->first,color_sym_group)),it->first));
    return non_symmetric_graphs;
}

int main()
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS,boost::undirectedS, boost::no_property, boost::property<alps::edge_type_t,alps::type_type> > graph_type;
//    unsigned int n_wo_sym = generate_graphs_without_symmetry<graph_type>(test_graph_size);
//    unsigned int n_w_sym = generate_graphs_with_edge_color_symmetry<graph_type>(test_graph_size);
    typedef alps::graph::graph_label<graph_type>::type label;
    std::map<label,graph_type> g_wo_sym = generate_graphs_without_symmetry_g<graph_type>(test_graph_size);
    std::map<label,graph_type> g_w_sym = generate_graphs_with_edge_color_symmetry_g<graph_type>(test_graph_size);
    unsigned int n_wo_sym = g_wo_sym.size();
    unsigned int n_w_sym  = g_w_sym.size();

    std::cout <<"w/o_sym:" << n_wo_sym << std::endl;
    std::cout <<"  w_sym:" << n_w_sym << std::endl;

    std::vector<label> r1;
    std::vector<label> r2;
    std::map<label,graph_type>::iterator it(g_wo_sym.begin()), end(g_wo_sym.end());
    for(; it != end; ++it)
        r1.push_back(it->first);
    it = g_w_sym.begin(), end = g_w_sym.end();
    for( ;it != end; ++it)
        r2.push_back(it->first);
    std::vector<label> difference;
    std::set_symmetric_difference(r1.begin(),r1.end(),r2.begin(),r2.end(),std::back_inserter(difference));

    for(std::size_t i=0; i < difference.size(); ++i)
    {
        std::cout << difference[i] << std::endl;
        if(g_wo_sym.find(difference[i]) != g_wo_sym.end())
        {
            graph_type g(g_wo_sym[difference[i]]);
            std::cout<< "wo_sym:"<< g << std::endl;
        }
        if(g_w_sym.find(difference[i]) != g_w_sym.end())
        {
            graph_type g(g_w_sym[difference[i]]);
            std::cout<< "w_sym:"<< g << std::endl;
        }
    }
    return n_wo_sym == n_w_sym ? 0 : 1;
}
