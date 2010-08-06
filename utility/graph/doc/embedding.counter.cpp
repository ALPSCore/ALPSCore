//[ embedding_counter_cpp
// file: embedding.counter.cpp
#include "embedding.lattice.hpp"
#include "embedding.counter.hpp"

#include <iostream>

#include <boost/graph/adjacency_list.hpp>

const std::size_t lattice_size = 12;
typedef boost::adjacency_list<
  boost::vecS, boost::vecS, boost::undirectedS
> graph_type;

int main() {
  // generate lattice
  graph_type lattice = create_lattice<graph_type>(lattice_size);
  // generate a small circle
  graph_type small(4);
  add_edge(0, 1, small);
  add_edge(1, 2, small);
  add_edge(2, 3, small);
  add_edge(3, 0, small);
  // create embedding counter
  embedding_counter<graph_type, graph_type> counter(small, lattice);
  // cout all possible embeddings of the small graph into the lattice
  std::cout << "Number of embeddings: " << counter.count() <<  std::endl;
  // count embedings if the vertices 0 and 0 are glued together
  std::cout << "Number of embeddings with 0 and 0 glued together: ";
  std::cout << counter.count(0, 0) <<  std::endl;
  return EXIT_SUCCESS;	
}
//]
