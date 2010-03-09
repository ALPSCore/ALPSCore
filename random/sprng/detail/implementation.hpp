/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <cstdlib>
#include <boost/throw_exception.hpp>
#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/parameter/macros.hpp>
#include <alps/random/parallel/detail/seed_macros.hpp>
#include <alps/random/sprng/detail/buffer.hpp>
#include <alps/random/sprng/keyword.hpp>

#if !defined(ALPS_SPRNG_GENERATOR)
#error Please set ALPS_SPRNG_GENERATOR to the name of one of the SPRNG generators before including this header file
#elif !defined(ALPS_SPRNG_TYPE)
#error Please set ALPS_SPRNG_TYPE to the type of one of the SPRNG generators before including this header file
#elif !defined(ALPS_SPRNG_MAX_STREAMS)
#error Please set ALPS_SPRNG_MAX_STREAMS to the maximum number of streams supported by the SPRNG generator
#elif !defined(ALPS_SPRNG_MAX_PARAMS)
#error Please set ALPS_SPRNG_MAX_PARAMS to the maximum number of parameters supported by the SPRNG generator
#else


#define ALPS_SPRNG_CALL(FUN) BOOST_PP_CAT(ALPS_SPRNG_GENERATOR,BOOST_PP_CAT(_,FUN))

#include <alps/random/sprng/detail/interface.hpp>

namespace alps { namespace random { namespace sprng {
using namespace alps::random::parallel;

/// wrapper for the 64-bit linear congruential generator of the SPRNG library
class ALPS_SPRNG_GENERATOR
{

public:
  /// SPRNG generators are all floating point generators at the moment
  typedef double result_type;
  enum { has_fixed_range = false};
  enum { sprng_type = ALPS_SPRNG_TYPE};
  
  /// the maximum number of independent streams
  enum { max_streams = ALPS_SPRNG_MAX_STREAMS };
  /// the number of different parameters for the constructor and seeding
  enum { max_param = ALPS_SPRNG_MAX_PARAMS };

  

  // we need custom copy constructor, destructor and assignment operators
  ALPS_SPRNG_GENERATOR(ALPS_SPRNG_GENERATOR const& rhs)
  {
    random::sprng::detail::buffer buf(rhs.sprng_ptr,&ALPS_SPRNG_CALL(pack_rng));
    sprng_ptr = buf.unpack(&ALPS_SPRNG_CALL(unpack_rng));
  }

  ~ALPS_SPRNG_GENERATOR()
  {
    free();
  }

  ALPS_SPRNG_GENERATOR const& operator=(ALPS_SPRNG_GENERATOR const& rhs)
  {
    free();
    detail::buffer buf(rhs.sprng_ptr,&ALPS_SPRNG_CALL(pack_rng));
    sprng_ptr = buf.unpack(&ALPS_SPRNG_CALL(unpack_rng));
    return *this;
  }

  /// seed function taking the SPRNG intialization arguments
  /// @param stream The number of this streamuence. Needs to be less than the num_stream parameter.
  /// @param num_stream The total number of random number streams to be created. Needs to be less than max_streams
  /// @param s the common seed for all random number streams
  /// @param param optional parametrization of the generator. Needs to be less than max_param.
  ///
  /// The SPRNG library guarantees that if the global_seed and param arguments are the same, 
  /// the random number streams specified by stream (0 <= stream < num_stream) are independent and non-overlapping.
  /// Changing eiher global_seed or the param will lead to different random number streams

  ALPS_RANDOM_PARALLEL_SEED_PARAMS(ALPS_SPRNG_GENERATOR,sprng_seed_params,: sprng_ptr(0))
  {
    unsigned int stream=p[stream_number|0u];
    unsigned int num_stream=p[total_streams|1u];
    int s=p[global_seed|0];
    unsigned int param=p[parameter|0u];

    BOOST_ASSERT(stream < num_stream);
    BOOST_ASSERT(num_stream < max_streams);
    BOOST_ASSERT(param < max_param);
    sprng_ptr= ALPS_SPRNG_CALL(init_rng) (sprng_type, stream, num_stream, s, param);
    if (sprng_ptr==0)
      boost::throw_exception(std::runtime_error("Failed initializing SPRNG generator"));
  }

  ALPS_RANDOM_PARALLEL_ITERATOR_SEED_DEFAULT()

  result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return 0.; }
  result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return 1.; }

  /// return the next value
  result_type operator()()
  {
    return ALPS_SPRNG_CALL(get_rn_dbl)(sprng_ptr);
  }

#ifdef ALPS_SPRNG_VALIDATION

  static bool validation(result_type x) 
  {
    return std::abs(x-ALPS_SPRNG_VALIDATION) < 1e-6;
  }

#endif

#ifdef BOOST_NO_OPERATORS_IN_NAMESPACE
    
  // Use a member function; Streamable concept not supported.
  bool operator==(const ALPS_SPRNG_GENERATOR& rhs) const
  {
    detail::buffer buf1(sprng_ptr,&ALPS_SPRNG_CALL(pack_rng));
    detail::buffer buf2(rhs.sprng_ptr,&ALPS_SPRNG_CALL(pack_rng));
    return buf1 == buf2;
  }
  bool operator!=(const ALPS_SPRNG_GENERATOR& rhs) const
  { return !(*this == rhs); }

#else 
  friend bool operator==(const ALPS_SPRNG_GENERATOR& x,
                         const ALPS_SPRNG_GENERATOR& y)
  { 
    detail::buffer buf1(x.sprng_ptr,&ALPS_SPRNG_CALL(pack_rng));
    detail::buffer buf2(y.sprng_ptr,&ALPS_SPRNG_CALL(pack_rng));
    return buf1 == buf2;
  }
  
  friend bool operator!=(const ALPS_SPRNG_GENERATOR& x,
                         const ALPS_SPRNG_GENERATOR& y)
  { return !(x == y); }
    
#if !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS) && !BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x551))
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os,
             const ALPS_SPRNG_GENERATOR& lcg)
  {
    detail::buffer buf(lcg.sprng_ptr,&ALPS_SPRNG_CALL(pack_rng));
    buf.write(os);
    return os;
  }

  template<class CharT, class Traits>
  friend std::basic_istream<CharT,Traits>&
  operator>>(std::basic_istream<CharT,Traits>& is,
             ALPS_SPRNG_GENERATOR& lcg)
  {
    detail::buffer buf;
    buf.read(is);
    lcg.sprng_ptr = buf.unpack(&ALPS_SPRNG_CALL(unpack_rng));
    return is;
  }
 
private:
#endif
#endif

  void free()
  {
    if(sprng_ptr)
      ALPS_SPRNG_CALL(free_rng(sprng_ptr));
    sprng_ptr=0;
  }
  
  int *sprng_ptr;    
};


#if defined(BOOST_NO_OPERATORS_IN_NAMESPACE) || defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS) || BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x551))
template<class CharT, class Traits>
std::basic_ostream<CharT,Traits>& operator<<(std::basic_ostream<CharT,Traits>& os, const ALPS_SPRNG_GENERATOR& lcg)
{
    detail::buffer buf(lcg.sprng_ptr,&ALPS_SPRNG_CALL(pack_rng));
    buf.write(os);
    return os;
}

template<class CharT, class Traits>
std::basic_istream<CharT,Traits>& operator>>(std::basic_istream<CharT,Traits>& is, ALPS_SPRNG_GENERATOR& lcg)
{
    detail::buffer buf;
    buf.read(is);
    lcg.sprng_ptr = buf.unpack(&ALPS_SPRNG_CALL(unpack_rng));
    return is;
}
#endif

} } } // namespace alps::random::sprng

#endif 

#undef ALPS_SPRNG_CALL
