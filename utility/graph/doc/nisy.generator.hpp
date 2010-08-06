//[ graph_generator_hpp
// file: nisy.generator.hpp
#ifndef NISY_GENERATOR_HPP
#define NISY_GENERATOR_HPP

#include "../src/nisy.hpp"

template <class graph_type> class graph_generator {
  typedef typename boost::graph_traits<
    graph_type
  >::vertex_iterator vertex_iterator;  
  public:
    // Iterator over all graphs with given #edges
  	typedef typename std::vector<graph_type>::const_iterator iterator;
    // Defaultconstructor
    graph_generator()
      : graphs_(1, std::vector<graph_type>(1, graph_type()))
    {
      add_vertex(graphs_[0][0]);
    }
    // generates all graphs with edge_size edges
    std::pair<iterator, iterator> generate(
      std::size_t edge_size
    ) {
      graph_type *graph, new_graph;
      typename std::vector<graph_type>::iterator it;
      vertex_iterator it1, end1, it2, end2;
      // construct all grophs with N edges from the Graphs with N-1 edges
      while (graphs_.size() <= edge_size) {
        graphs_.push_back(std::vector<graph_type>());
        it = (graphs_.end() - 2)->begin(); 
        while ((graph=(it==(graphs_.end()-2)->end()?NULL:&(*it++)))!=NULL)
          for (tie(it1, end1) = vertices(*graph); it1 != end1; ++it1) {
            new_graph = graph_type(*graph);
            add_edge(*it1, add_vertex(new_graph), new_graph);
            check_graph(new_graph);
            for (tie(it2, end2) = vertices(*graph); it2 != end2; ++it2)
              if (*it1 < *it2 && !edge(*it1, *it2, *graph).second) {
                new_graph = graph_type(*graph);
                add_edge(*it1, *it2, new_graph);
	            check_graph(new_graph);
            }
          }
      }
      return std::make_pair(
        graphs_[edge_size].begin(), graphs_[edge_size].end()
      );
    }
  private:
    void check_graph(graph_type const & graph) {
      // create comparable adapter
      nisy<graph_type> com_graph(graph);
      // check if the label has alredy been found
      if (labels_.find(make_tuple(com_graph.get_canonical_label().size(), com_graph.get_canonical_label())) == labels_.end()) {
        labels_.insert(make_tuple(com_graph.get_canonical_label().size(), com_graph.get_canonical_label()));
        graphs_.back().push_back(graph);
      }
    }
    std::vector<std::vector<graph_type> > graphs_;
    // list of all lables already checked.
    std::set<boost::tuple<std::size_t, typename nisy<graph_type>::canonical_label_type> > labels_;
};

#endif // NISY_GENERATOR_HPP
//]
