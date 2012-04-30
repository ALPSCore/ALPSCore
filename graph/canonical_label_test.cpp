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

#include <alps/graph/canonical_properties.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>


using boost::get;
using alps::graph::canonical_properties;

bool colored_edges_test()
{
    typedef boost::property<alps::edge_type_t,unsigned int> edge_props;
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, edge_props> graph_type;
    typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor;
    typedef boost::property_map<graph_type,alps::edge_type_t>::type edge_color_map_type;

    graph_type g(4);
    edge_color_map_type g_edge_color = get(alps::edge_type_t(),g);
    edge_descriptor e = add_edge(0, 1, g).first;
    boost::put(g_edge_color,e,0);
    e = add_edge(1, 2, g).first;
    boost::put(g_edge_color,e,0);
    e = add_edge(2, 3, g).first;
    boost::put(g_edge_color,e,0);
    e = add_edge(0, 3, g).first;
    boost::put(g_edge_color,e,0);

    graph_type h(g);
    edge_color_map_type h_edge_color = get(alps::edge_type_t(),h);
    e = edge(2,3,h).first;
    boost::put(h_edge_color,e,1);
    
    graph_type i(g);
    edge_color_map_type i_edge_color = get(alps::edge_type_t(),i);
    e = edge(2,3,i).first;
    boost::put(i_edge_color,e,2);

    graph_type j(h);
    edge_color_map_type j_edge_color = get(alps::edge_type_t(),j);
    e = edge(2,3,j).first;
    boost::put(j_edge_color,e,1);
  

    alps::graph::graph_label<graph_type>::type label_g(get<1>(canonical_properties(g)));
    alps::graph::graph_label<graph_type>::type label_h(get<1>(canonical_properties(h)));
    alps::graph::graph_label<graph_type>::type label_i(get<1>(canonical_properties(i)));
    alps::graph::graph_label<graph_type>::type label_j(get<1>(canonical_properties(j)));

    std::cout<<label_g<<std::endl;
    std::cout<<label_h<<std::endl;
    std::cout<<label_i<<std::endl;
    std::cout<<label_j<<std::endl;
    std::cout<<std::boolalpha<<(label_g == label_h)<<std::endl;
    std::cout<<std::boolalpha<<(label_h == label_i)<<std::endl;
    std::cout<<std::boolalpha<<(label_h == label_j)<<std::endl;

    return true;
}

bool simple_test() {
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;
    graph_type g;
    add_edge(0, 1,g);
    add_edge(0, 2,g);
    add_edge(0, 3,g);
    add_edge(2, 1,g);

    alps::graph::graph_label<graph_type>::type label(get<1>(canonical_properties(g)));
    
    std::cout << label << std::endl;
    return true;
}

int main() {
    simple_test();

    colored_edges_test();

    return 0;
}
