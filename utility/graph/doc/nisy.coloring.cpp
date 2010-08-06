//[ coloring_cpp
// file: coloring.cpp
#include "../src/nisy.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map.hpp>
#include <iostream>

typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS
> graph_type;

typedef std::map<
    boost::graph_traits<graph_type>::vertex_descriptor
  , int
> map_type;
typedef boost::associative_property_map<map_type> property_map_type;
enum { A, B, C, D, N };

typedef nisy<graph_type>::cell_type cell_type;
typedef nisy<graph_type>::partition_type partition_type;

// Write a ordering to cout
template<class ordering_iterator> void dump_ordering(std::pair<ordering_iterator, ordering_iterator> const & P) {
  std::cout << "{";
  for (ordering_iterator it = P.first; it != P.second; ++it)
    std::cout << " " << *it;
  std::cout << " }" << std::endl;
}

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

  // create coloring
  map_type map_g, map_h;
  property_map_type pmap_g(map_g), pmap_h(map_h);
  
  // create coloring of g
  boost::put(pmap_g, static_cast<int>(A), 0);
  boost::put(pmap_g, static_cast<int>(B), 0);
  boost::put(pmap_g, static_cast<int>(C), 1);
  boost::put(pmap_g, static_cast<int>(D), 1);

  // create coloring of h
  boost::put(pmap_h, static_cast<int>(A), 0);
  boost::put(pmap_h, static_cast<int>(B), 1);
  boost::put(pmap_h, static_cast<int>(C), 0);
  boost::put(pmap_h, static_cast<int>(D), 1);
  
  // create comparable adaptors
  nisy<graph_type, int> cg(g, pmap_g), ch(h, pmap_h);

  // check if the graphs g and h are isomorphic
  if (cg == ch)
    std::cout << "The two graphs are isomorphic." << std::endl;
  else
    std::cout << "The two graphs are not isomorphic." << std::endl;

  // dump canonical partiton and arbit of g
  std::cout << std::endl;
  std::cout << "Canonical partiton and orbit partition of G" << std::endl;
  dump_ordering(cg.get_canonical_ordering());
  dump_partition(cg.get_orbit_partition());
  
  // dump canonical partition and arbit of h
  std::cout << std::endl;
  std::cout << "Canonical partition and orbit partition of H" << std::endl;
  dump_ordering(ch.get_canonical_ordering());
  dump_partition(ch.get_orbit_partition());

  // compute the isomprphism between the two grphas
  typedef std::map<
  	boost::graph_traits<graph_type>::vertex_descriptor, 
  	boost::graph_traits<graph_type>::vertex_descriptor 
  > isomorphism_type;
  isomorphism_type iso(isomorphism(cg, ch));

  // write the isomprphism to cout
  std::cout << std::endl << "Isomorphism G => H" << std::endl;
  for (isomorphism_type::iterator it = iso.begin(); it != iso.end(); ++it)
  	std::cout << "(" << it->first << "->" << it->second << ") ";
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
//]
