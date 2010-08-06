//[ embedding_lattice_hpp
// file: embedding.lattice.hpp
#ifndef EMBEDDING_LATTICE_HPP
#define EMBEDDING_LATTICE_HPP

#include <boost/graph/adjacency_list.hpp>

// create a sqare lattice of size L x L
template <class graph_type> graph_type create_lattice(std::size_t L) {
  graph_type lattice(L * L);
  for (std::size_t i = 0; i < L; ++i)
    for (std::size_t j = 0; j < L; ++j) {
      add_edge(i * L + j, (i * L + j + 1) % (L * L), lattice);
      add_edge(i * L + j, ((i + 1) * L + j) % (L * L), lattice);
    }
  return lattice;
}

#endif // EMBEDDING_LATTICE_HPP
//]
