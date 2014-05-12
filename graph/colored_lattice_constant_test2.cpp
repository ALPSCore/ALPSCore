/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2012 by Andreas Hehn <hehn@phys.ethz.ch>                          *
 *                       Lukas Gamper <gamperl@gmail.com>                          *
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

#include <alps/graph/lattice_constant.hpp>
#include <alps/lattice/lattice.h>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>
int main() {
    using boost::get;
    using boost::put;
    using alps::graph::canonical_properties;

    typedef unsigned int lc_type;
    
    typedef boost::property<alps::edge_type_t,alps::type_type> edge_props;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,boost::no_property,edge_props> graph_type;
    typedef alps::graph_helper<>::vertex_iterator vertex_iterator;
    typedef alps::graph_helper<>::edge_iterator edge_iterator;

    alps::Parameters parm;
    unsigned int side_length = 40;
    
    std::ifstream in("../../lib/xml/lattices.xml");
    parm["LATTICE"] = "anisotropic square lattice";
    parm["L"] = side_length;
    alps::graph_helper<> lattice(in,parm);
    
    graph_type lattice_graph(num_vertices(lattice.graph()));
    boost::graph_traits<alps::graph_helper<>::graph_type>::edge_iterator it, et;
    for(boost::tie(it, et) = edges(lattice.graph()); it != et; ++it)
    {
        boost::graph_traits<graph_type>::edge_descriptor  e = add_edge(source(*it, lattice.graph()), target(*it, lattice.graph()), lattice_graph).first;
        put(alps::edge_type_t(), lattice_graph, e, get(alps::edge_type_t(),lattice.graph(),*it) );
    }

    std::vector<std::pair<graph_type,lc_type> > g;
    boost::graph_traits<graph_type>::edge_descriptor e;

    // edge color 0 ...
    // edge color 1 ___
    
    //
    //  0...1
    //  |   |
    //  2...3
    //
    g.push_back(std::make_pair(graph_type(), 1));
    e = add_edge(0, 1, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(1, 3, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(3, 2, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(2, 0, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    
    //
    //  0...1
    //  .   |
    //  2___3
    //
    g.push_back(std::make_pair(graph_type(), 0));
    e = add_edge(0, 1, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(1, 3, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(3, 2, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);
    e = add_edge(2, 0, g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);



    //
    //  1___0___2
    //
    g.push_back(std::make_pair(graph_type(),1));
    e = add_edge(0, 1,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(0, 2,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    
    //
    //  1...0___2
    //
    g.push_back(std::make_pair(graph_type(),4));
    e = add_edge(0, 1,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 0);
    e = add_edge(0, 2,g.back().first).first;
    put(alps::edge_type_t(), g.back().first, e, 1);

    int success = 0;
    for(std::vector<std::pair<graph_type, lc_type> >::iterator it= g.begin(); it != g.end(); ++it)
    {
        lc_type lc = alps::graph::lattice_constant(
              it->first
            , lattice_graph
            , lattice.lattice()
            , alps::cell(std::vector<int>(2,side_length/2),lattice.lattice()) //side_length * side_length / 2 + side_length / 2 - 1
        );
        if ( lc != it->second)
        {
            std::cerr<<"ERROR: lattice constant does not match!"<<std::endl;
            std::cerr<<"Graph:"<<std::distance(g.begin(),it)<<" Calculated: "<<lc<<"\tReference: "<<it->second<<std::endl<<std::endl;
            success = -1;
        }
    }
    return success;
}
