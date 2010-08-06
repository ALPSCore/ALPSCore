//[ graph_generator_cpp
// file: nisy.generator.cpp
#include "nisy.generator.hpp"

#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/graph/adjacency_list.hpp>

#define MAX_GRAPH_SIZE 12
typedef boost::adjacency_list<
  boost::vecS, boost::vecS, boost::undirectedS
> graph_type;

int main() {
  // construct generator
  graph_generator<graph_type> generator;
  graph_generator<graph_type>::iterator it, end;
  boost::timer timer;
  std::cout << "#edges #graphs time[s]" << std::endl;
  for (std::size_t i = 0; i <= MAX_GRAPH_SIZE; ++i) {
    // generate the graphs
    boost::tie(it, end) = generator.generate(i);
    // write the result to cout
    std::cout << std::setw(6) << i << " "
              << std::setw(7) << (end - it) << " " 
              << std::setw(7)
              << static_cast<std::size_t>(timer.elapsed()+.5) 
              << std::endl;
  }
  return EXIT_SUCCESS;	
}
//]
