/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

/// \file keyword.hpp
/// \brief This file defines the named parameter keywords for the random library

#include <boost/parameter/keyword.hpp>
#include <boost/parameter/parameters.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/mpl/placeholders.hpp>

#ifndef ALPS_RANDOM_PARALLEL_KEYWORD_HPP
#define ALPS_RANDOM_PARALLEL_KEYWORD_HPP

#ifndef ALPS_RANDOM_MAXARITY
/// The number of named parameter arguments for the random library. The default value can be changed by defining the macro.
#define ALPS_RANDOM_MAXARITY 5
#endif


namespace alps { namespace random { namespace parallel {
  using boost::mpl::placeholders::_;
  
  BOOST_PARAMETER_KEYWORD(random_tag,global_seed)
  BOOST_PARAMETER_KEYWORD(random_tag,stream_number)
  BOOST_PARAMETER_KEYWORD(random_tag,total_streams)
  BOOST_PARAMETER_KEYWORD(random_tag,first)
  BOOST_PARAMETER_KEYWORD(random_tag,last)

  /// INTERNAL ONLY
  typedef boost::parameter::parameters<
      boost::parameter::optional<random_tag::global_seed>
    , boost::parameter::optional<random_tag::stream_number, boost::is_convertible<_,unsigned int> >
    , boost::parameter::optional<random_tag::total_streams, boost::is_convertible<_,unsigned int> >
  > seed_params;

  /// INTERNAL ONLY
  typedef boost::parameter::parameters<
      boost::parameter::required<random_tag::first>
    , boost::parameter::required<random_tag::last>
    , boost::parameter::optional<random_tag::stream_number, boost::is_convertible<_,unsigned int> >
    , boost::parameter::optional<random_tag::total_streams, boost::is_convertible<_,unsigned int> >
  > iterator_seed_params;

} } } // namespace alps::random::parallel

#endif // ALPS_RANDOM_PARALLEL_KEYWORD_HPP
