//[ embedding_list_cpp
// file: embedding.lsit.cpp
#include "embedding.counter.hpp"
#include "nisy.generator.hpp"
#include "embedding.lattice.hpp"

#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/graph/adjacency_list.hpp>

#define MAX_GRAPH_SIZE 12
#define LATTICE_SIZE 12
typedef boost::adjacency_list<
  boost::vecS, boost::vecS, boost::undirectedS
> graph_type;

int main() {
  // generate lattice
  graph_type lattice = create_lattice<graph_type>(LATTICE_SIZE);
  // construct generator
  graph_generator<graph_type> generator;
  graph_generator<graph_type>::iterator it, end;
  boost::timer timer;
  std::cout << "#edges #graphs #embeddings time[s]" << std::endl;
  for (std::size_t i = 0; i <= MAX_GRAPH_SIZE; ++i) {
    // generate the graphs
    boost::tie(it, end) = generator.generate(i);
    std::size_t gr_cnt = 0, emb_cnt = 0;
    for(; it != end; it++) {
      // create an embedding counter
      std::size_t cnt = embedding_counter<
        graph_type, graph_type
      >(*it, lattice).count(0, 0);
      emb_cnt += cnt;
      if (cnt)
      	 ++gr_cnt;
    }
    // write the result to cout
    std::cout << std::setw(6) << i << " "
              << std::setw(7) << gr_cnt << " " 
              << std::setw(11) << emb_cnt  << " "
              << std::setw(7) 
              << static_cast<std::size_t>(timer.elapsed() + .5) 
              << std::endl;
  }
  return EXIT_SUCCESS;
}
//]
