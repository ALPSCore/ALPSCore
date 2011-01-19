#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <alps/graph/nisy.hpp>

//
// Workarounds to pass-through the !*#$@$%# property maps for the !*#$@$%# boost graphs...
//
template <typename PropertyBundle>
struct bundle_color_type
{
    typedef typename PropertyBundle::color_type type;
};

template <>
struct bundle_color_type<boost::no_bundle>
{
    typedef void type;
};

template <typename Graph, typename VertexPropertyBundle, typename EdgePropertyBundle>
struct nisy_wrapper : public  nisy<Graph, typename bundle_color_type<VertexPropertyBundle>::type, typename bundle_color_type<EdgePropertyBundle>::type >
{
    nisy_wrapper(Graph const& g)
    : nisy<Graph, typename bundle_color_type<VertexPropertyBundle>::type, typename bundle_color_type<EdgePropertyBundle>::type >
      (g,get(&VertexPropertyBundle::color,g), get(&EdgePropertyBundle::color, g))
    {}
};

template <typename Graph, typename VertexPropertyBundle>
struct nisy_wrapper<Graph,VertexPropertyBundle,boost::no_bundle> : public nisy<Graph, typename bundle_color_type<VertexPropertyBundle>::type, void >
{
    nisy_wrapper(Graph const& g)
    : nisy<Graph, typename bundle_color_type<VertexPropertyBundle>::type, void >
      (g,get(&VertexPropertyBundle::color,g))
    {}
};

template <typename Graph, typename EdgePropertyBundle>
struct nisy_wrapper<Graph,boost::no_bundle,EdgePropertyBundle> : public nisy<Graph, void, typename bundle_color_type<EdgePropertyBundle>::type>
{
    nisy_wrapper(Graph const& g)
    : nisy<Graph, void, typename bundle_color_type<EdgePropertyBundle>::type>
      (g,get(&EdgePropertyBundle::color,g))
    {}
};



//
// The graph generator class (finally...)
//
template <class graph_type>
class graph_generator {
    typedef typename boost::graph_traits< graph_type >::vertex_iterator vertex_iterator;  
    typedef typename boost::vertex_bundle_type<graph_type>::type        vertex_property_bundle_type;
    typedef typename boost::edge_bundle_type<graph_type>::type          edge_property_bundle_type;
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
    std::pair<iterator, iterator> generate( std::size_t edge_size) {
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
      nisy_wrapper<graph_type, vertex_property_bundle_type, edge_property_bundle_type > com_graph(graph);
      // check if the label has alredy been found
      if (labels_.find(com_graph.get_canonical_label()) == labels_.end()) {
        labels_.insert(com_graph.get_canonical_label());
        graphs_.back().push_back(graph);
      }
    }
    std::vector<std::vector<graph_type> > graphs_;
    // list of all lables already checked.
    std::set<typename nisy_wrapper< graph_type, vertex_property_bundle_type, edge_property_bundle_type >::canonical_label_type > labels_;
};

template <typename graph_type>
void dump_graph(graph_type const& g)
{
    std::cout<<"graph g {"<<std::endl;
    typename boost::graph_traits<graph_type>::edge_iterator it, end;
    for(boost::tie(it,end) =edges(g); it != end; ++it)
        std::cout<< source(*it,g) <<" -- "<< target(*it,g) <<";"<<std::endl;
    std::cout<<"}"<<std::endl<<std::endl;
}

struct bond_type
{
    typedef int color_type;
    color_type color;

    bool operator != (bond_type const& rhs)
    {
        return color != rhs.color;
    }
};

int main()
{

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, bond_type> graph_type;
    graph_generator<graph_type> graph_gen;

    graph_generator<graph_type>::iterator it,end;
    for(boost::tie(it,end) = graph_gen.generate(6); it != end; ++it)
        dump_graph(*it);

    return 0;
}
