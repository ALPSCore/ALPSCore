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

#include <alps/graph/canonical_properties.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <iostream>
#include <iterator>

typedef boost::adjacency_list<
  boost::vecS, boost::vecS, boost::undirectedS
> graph_type;

typedef boost::adjacency_list<
	  boost::vecS
	, boost::vecS
	, boost::undirectedS
	, boost::property<boost::vertex_name_t, std::size_t>
	, boost::property<boost::edge_name_t, std::size_t>
> colored_graph_type;

using namespace alps::graph;

// Write a partition to cout
template<class Partition> void dump_partition(Partition const & P) {
  std::cout << "{";
  typename Partition::const_iterator it1;
  typename Partition::value_type::const_iterator it2;
  for (it1 = P.begin(); it1 != P.end(); ++it1) {
    std::cout << "(";
    for (it2 = it1->begin(); it2 != it1->end(); ++it2)
      std::cout << (it2 == it1->begin() ? "" : " ") << *it2;
    std::cout << ")";
  }
  std::cout << "}" << std::endl;
}

int main() {

	{
		enum { A, B, C, D, N };

		colored_graph_type g(N), h(N);
		
		/*
			A - B       A   B
			| / |  vs.  | X |
			C - D       C - D
		*/

		add_edge(A, B, g);
		add_edge(A, C, g);
		add_edge(B, C, g);
		add_edge(B, D, g);
		add_edge(C, D, g);

		add_edge(A, C, h);
		add_edge(A, D, h);
		add_edge(B, C, h);
		add_edge(B, D, h);
		add_edge(C, D, h);

		boost::property_map<colored_graph_type, boost::vertex_name_t>::type g_vertex_name = get(boost::vertex_name_t(), g);
		g_vertex_name[A] = 0;
		g_vertex_name[B] = 1;
		g_vertex_name[C] = 1;
		g_vertex_name[D] = 0;

		boost::property_map<colored_graph_type, boost::vertex_name_t>::type h_vertex_name = get(boost::vertex_name_t(), h);
		h_vertex_name[A] = 0;
		h_vertex_name[B] = 0;
		h_vertex_name[C] = 1;
		h_vertex_name[D] = 1;
		
		typename boost::graph_traits<colored_graph_type>::edge_iterator it, end;

		boost::property_map<colored_graph_type, boost::edge_name_t>::type g_edge_name = get(boost::edge_name_t(), g);
		for (boost::tie(it, end) = edges(g); it != end; ++it)
			g_edge_name[*it] = (source(*it, g) == B && target(*it, g) == C) ? 1 : 0;

		boost::property_map<colored_graph_type, boost::edge_name_t>::type h_edge_name = get(boost::edge_name_t(), h);
		for (boost::tie(it, end) = edges(h); it != end; ++it)
			h_edge_name[*it] = (source(*it, h) == C && target(*it, h) == D) ? 1 : 0;

		std::vector<boost::graph_traits<colored_graph_type>::vertex_descriptor> g_ordering, h_ordering;
		graph_label<colored_graph_type>::type g_label, h_label;
		partition_type<colored_graph_type>::type g_orbit, h_orbit;
		boost::tie(g_ordering, g_label, g_orbit) = canonical_properties(g);
		boost::tie(h_ordering, h_label, h_orbit) = canonical_properties(h);

		for (std::vector<boost::graph_traits<colored_graph_type>::vertex_descriptor>::const_iterator it = g_ordering.begin(); it != g_ordering.end(); ++it)
			std::cout << (it != g_ordering.begin() ? " " : "(") << *it;
		std::cout << ")" << std::endl;
		for (std::vector<boost::graph_traits<colored_graph_type>::vertex_descriptor>::const_iterator it = h_ordering.begin(); it != h_ordering.end(); ++it)
			std::cout << (it != h_ordering.begin() ? " " : "(") << *it;
		std::cout << ")" << std::endl;

		dump_partition(g_orbit);
		dump_partition(h_orbit);
		
		std::cout << g_label << std::endl;
		std::cout << h_label << std::endl;
	}

	{
		enum {A, B, C, D, E, F, G, H, I, N};

		graph_type g(N);
		
		/*
			A - B - C
			|   |   |
			D - E - F
			|   |   |
			G - H - I
		*/

		add_edge(A, B, g);
		add_edge(B, C, g);
		
		add_edge(A, D, g);
		add_edge(B, E, g);
		add_edge(C, F, g);
		
		add_edge(D, E, g);
		add_edge(E, F, g);
		
		add_edge(D, G, g);
		add_edge(E, H, g);
		add_edge(F, I, g);
		
		add_edge(G, H, g);
		add_edge(H, I, g);

		std::vector<boost::graph_traits<graph_type>::vertex_descriptor> g_ordering;
		graph_label<graph_type>::type g_label;
		partition_type<graph_type>::type g_orbit;
		boost::tie(g_ordering, g_label, g_orbit) = canonical_properties(g);
		
		for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = g_ordering.begin(); it != g_ordering.end(); ++it)
			std::cout << (it != g_ordering.begin() ? " " : "(") << *it;
		std::cout << ")" << std::endl;
		dump_partition(g_orbit);
		std::cout << g_label << std::endl;
		
	}

	{
		enum { A, B, C, D, N };

		graph_type g(N), h(N);
		
		/*
			A - B       A   B
			| / |  vs.  | X |
			C - D       C - D
		*/
		
		add_edge(A, B, g);
		add_edge(A, C, g);
		add_edge(B, C, g);
		add_edge(B, D, g);
		add_edge(C, D, g);

		add_edge(A, C, h);
		add_edge(A, D, h);
		add_edge(B, C, h);
		add_edge(B, D, h);
		add_edge(C, D, h);

		std::vector<boost::graph_traits<graph_type>::vertex_descriptor> g_ordering, h_ordering;
		graph_label<graph_type>::type g_label, h_label;
		partition_type<graph_type>::type g_orbit, h_orbit;
		boost::tie(g_ordering, g_label, g_orbit) = canonical_properties(g);
		boost::tie(h_ordering, h_label, h_orbit) = canonical_properties(h);

		for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = g_ordering.begin(); it != g_ordering.end(); ++it)
			std::cout << (it != g_ordering.begin() ? " " : "(") << *it;
		std::cout << ")" << std::endl;
		for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = h_ordering.begin(); it != h_ordering.end(); ++it)
			std::cout << (it != h_ordering.begin() ? " " : "(") << *it;
		std::cout << ")" << std::endl;

		dump_partition(g_orbit);
		dump_partition(h_orbit);
		
		std::cout << g_label << std::endl;
		std::cout << h_label << std::endl;
	}

    {
        /*
                 F
                 |
              E--A--B
                / \
               D   C

                vs.

                  D
                  |
               A--B--C--E--F

                vs.

                   D
                   |
                A--B--C--F
                   |
                   E
        */
        enum { A, B, C, D, E, F, N};

        graph_type g(N), h(N), i(N);

        add_edge(A,B,g);
        add_edge(A,C,g);
        add_edge(A,D,g);
        add_edge(A,E,g);
        add_edge(A,F,g);

        add_edge(A,B,h);
        add_edge(B,C,h);
        add_edge(B,D,h);
        add_edge(C,E,h);
        add_edge(E,F,h);

        add_edge(A,B,i);
        add_edge(B,C,i);
        add_edge(B,D,i);
        add_edge(B,E,i);
        add_edge(C,F,i);
		
        std::vector<boost::graph_traits<graph_type>::vertex_descriptor> g_ordering, h_ordering, i_ordering;
		graph_label<graph_type>::type g_label, h_label, i_label;
		partition_type<graph_type>::type g_orbit, h_orbit, i_orbit;
		boost::tie(g_ordering, g_label, g_orbit) = canonical_properties(g);
		boost::tie(h_ordering, h_label, h_orbit) = canonical_properties(h);
		boost::tie(i_ordering, i_label, i_orbit) = canonical_properties(i);

		for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = g_ordering.begin(); it != g_ordering.end(); ++it)
			std::cout << (it != g_ordering.begin() ? " " : "(") << *it;
		std::cout << ")" << std::endl;
		for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = h_ordering.begin(); it != h_ordering.end(); ++it)
			std::cout << (it != h_ordering.begin() ? " " : "(") << *it;
		std::cout << ")" << std::endl;
		for (std::vector<boost::graph_traits<graph_type>::vertex_descriptor>::const_iterator it = i_ordering.begin(); it != i_ordering.end(); ++it)
			std::cout << (it != i_ordering.begin() ? " " : "(") << *it;
		std::cout << ")" << std::endl;

		dump_partition(g_orbit);
		dump_partition(h_orbit);
		dump_partition(i_orbit);
		
		std::cout << g_label << std::endl;
		std::cout << h_label << std::endl;
		std::cout << i_label << std::endl;
    }

	return EXIT_SUCCESS;	
}
