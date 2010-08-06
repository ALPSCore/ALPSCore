//[ nauty_simple
// file: nauty.simple.cpp
#include "../src/nauty.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <iostream>
#include <iterator>

typedef boost::adjacency_list<
  boost::vecS, boost::vecS, boost::undirectedS
> graph_type;
enum { A, B, C, D, N };

// Write a partition to cout
template<class partition_type> void dump_partition(partition_type const & P) {
  std::cout << "{";
  typename partition_type::const_iterator it1;
  typename partition_type::value_type::const_iterator it2;
  for (it1 = P.begin(); it1 != P.end(); ++it1) {
    std::cout << "(";
    for (it2 = it1->begin(); it2 != it1->end(); ++it2)
      std::cout << " " << *it2;
    std::cout << " )";
  }
  std::cout << "}" << std::endl;
}

int main() {

  // create the original graphs
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
  
  // create comparable adaptors
  nauty<graph_type> ng(g), nh(h);

  // check if the graphs g and h are isomorphic
  if (ng == nh)
    std::cout << "The two graphs are isomorphic." << std::endl;
  else
    std::cout << "The two graphs are not isomorphic." << std::endl;
 
  // dump canonical partiton and arbit of g
  std::cout << std::endl;
  std::cout << "Canonical partiton and orbit partition of G" << std::endl;
  std::copy(
      ng.get_canonical_ordering().first
    , ng.get_canonical_ordering().second
    , std::ostream_iterator<boost::graph_traits<graph_type>::vertex_descriptor>(std::cout, " ")
  );
  std::cout << std::endl;
  dump_partition(ng.get_orbit_partition());
  
  // dump canonical partition and arbit of h
  std::cout << std::endl;
  std::cout << "Canonical partition and orbit partition of H" << std::endl;
  std::copy(
      nh.get_canonical_ordering().first
    , nh.get_canonical_ordering().second
    , std::ostream_iterator<boost::graph_traits<graph_type>::vertex_descriptor>(std::cout, " ")
  );
  std::cout << std::endl;
  dump_partition(nh.get_orbit_partition());
   
  // compute the isomprphism between the two grphas
  typedef std::map<
  	boost::graph_traits<graph_type>::vertex_descriptor, 
  	boost::graph_traits<graph_type>::vertex_descriptor 
  > isomorphism_type;
  isomorphism_type iso(isomorphism(ng, nh));
  
  // write the isomprphism to cout
  std::cout << std::endl << "Isomorphism G => H" << std::endl;
  for (isomorphism_type::iterator it = iso.begin(); it != iso.end(); ++it)
  	std::cout << "(" << it->first << "->" << it->second << ") ";
  std::cout << std::endl;

  return EXIT_SUCCESS;	
}

//]
