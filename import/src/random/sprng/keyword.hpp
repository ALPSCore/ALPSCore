/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

#include <alps/random/parallel/keyword.hpp>

#ifndef ALPS_RANDOM_SPRNG_KEYWORD_HPP
#define ALPS_RANDOM_SPRNG_KEYWORD_HPP

namespace alps { namespace random { namespace parallel {
  using boost::mpl::placeholders::_;
  
  BOOST_PARAMETER_KEYWORD(random_tag,parameter)

  /// INTERNAL ONLY
  typedef boost::parameter::parameters<
      boost::parameter::optional<random_tag::stream_number, boost::is_convertible<_,unsigned int> >
    , boost::parameter::optional<random_tag::total_streams, boost::is_convertible<_,unsigned int> >
    , boost::parameter::optional<random_tag::global_seed, boost::is_convertible<_,int> >
    , boost::parameter::optional<random_tag::parameter, boost::is_convertible<_,unsigned int> >
  > sprng_seed_params;

} } } // namespace alps::random::parallel

#endif // ALPS_RANDOM_SPRNG_KEYWORD_HPP
