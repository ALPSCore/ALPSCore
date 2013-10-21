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

static unsigned int const test_graph_size = 7;

template <typename Graph>
void subgraph_generator_with_color_symmetries_test(unsigned int order)
{
    std::ifstream in("../../lib/xml/lattices.xml");
    alps::Parameters parm;
    parm["LATTICE"] = "anisotropic triangular lattice";
    parm["L"]       = 2*order+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef alps::coordinate_graph_type lattice_graph_type;
    lattice_graph_type& lattice_graph = alps_lattice.graph();

    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type,alps::graph::policies::canonical_properties_with_color_symmetries_policy<Graph> > graph_gen_type;
    graph_gen_type graph_gen(lattice_graph, 2*order*order);

    typename alps::graph::color_partition<Graph>::type color_sym_group;
    color_sym_group[0] = 0;
    color_sym_group[1] = 0;
    color_sym_group[2] = 0;
    graph_gen.set_color_partition(color_sym_group);

    typename graph_gen_type::iterator it,end;

    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order);
    std::cout<< std::distance(it,end) << std::endl;
}

int main()
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS,boost::undirectedS, boost::no_property, boost::property<alps::edge_type_t,alps::type_type> > graph_type;
    subgraph_generator_with_color_symmetries_test<graph_type>(test_graph_size);
    return 0;
}
