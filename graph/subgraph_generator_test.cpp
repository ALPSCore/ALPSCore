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

//#define USE_COMPRESSED_EMBEDDING2

#include <alps/graph/subgraph_generator.hpp>
#include <alps/lattice/graph_helper.h>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

#include <boost/timer/timer.hpp>


enum { test_graph_size = 10 };

template <typename Graph>
void subgraph_generator_test(unsigned int order_ )
{
    std::ifstream in("../../lib/xml/lattices.xml");
    alps::Parameters parm;
    parm["LATTICE"] = "square lattice";
    parm["L"] = 2*order_+1;
    alps::graph_helper<> alps_lattice(in,parm);

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> lattice_graph_type;
    boost::graph_traits<alps::graph_helper<>::graph_type>::edge_iterator eit, eend;
	lattice_graph_type lattice_graph(num_vertices(alps_lattice.graph()));
    for(boost::tie(eit,eend) = edges(alps_lattice.graph()); eit != eend; ++eit)
        add_edge( static_cast<unsigned int>(source(*eit,alps_lattice.graph()))
                , static_cast<unsigned int>(target(*eit,alps_lattice.graph()))
                , lattice_graph
                );
    
    typedef alps::graph::subgraph_generator<Graph,lattice_graph_type> graph_gen_type;
    graph_gen_type graph_gen(lattice_graph,2*order_*order_);
    
    typename graph_gen_type::iterator it,end;
//    boost::timer::auto_cpu_timer t;
    boost::tie(it,end) = graph_gen.generate_up_to_n_edges(order_);
    std::cout<<std::distance(it,end)<<std::endl;
}

int main()
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS,boost::undirectedS> graph_type;
    subgraph_generator_test<graph_type>(test_graph_size);
    return 0;
}
