#ifndef NAUTY_HPP
#define NAUTY_HPP

#include "nisy.hpp"

template <
    class graph_type
> class nauty 
  : public nisy<graph_type> 
{
  public:
    nauty (
        graph_type const & graph
    )
      : nisy<graph_type>(graph)
    {}
};

#endif // NAUTY_HPP
