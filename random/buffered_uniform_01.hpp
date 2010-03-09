/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#ifndef ALPS_RANDOM_BUFFERED_UNIFORM_01_HPP
#define ALPS_RANDOM_BUFFERED_UNIFORM_01_HPP

/// @file This file declares and implements a polymorphic buffered random
/// number generator creating floating point random nubers in the interval [0,1)


#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>
#include <alps/random/buffered_generator.hpp>

#include <iostream>

namespace alps {

/// @brief the abstract base class of a runtime-polymorphic buffered random number generator 
/// generating double values in the interval [0,1[
///
/// @tparam RealType the floating point type of the random numbers, defaults to double
///
/// This class template is an abstract base class template for runtime-polymorphic uniform random
/// number generators producing numbers in the interval [0,1). It inherits from 
/// @ref alps::buffered_generator and provides @c min() and @max () functions returning 0. and 1. respectively, 
// which makes this class template a model of UniformRandomNumberGenerator.


template <class RealType=double> 
class buffered_uniform_01 
 : public buffered_generator<RealType>
{
public:

  /// the type of random numbers
  typedef RealType result_type;
  
  /// constructs the generator
  /// @param buffer_size the size of the buffer

  buffered_uniform_01(std::size_t buffer_size=ALPS_BUFFERED_GENERATOR_BUFFER_SIZE) 
   :  buffered_generator<result_type>(buffer_size)
  {}

#ifdef ALPS_DOXYGEN
  /// @returns 0.
  result_type min() const;
  /// @returns 1.
  result_type max() const;
#else
  result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return static_cast<RealType>(0.); }
  result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return static_cast<RealType>(1.); }
#endif
};


/// a runtime-polymorphic buffered random number generator 
/// generating double values in the interval [0,1[
///
/// @tparam Engine the type of random number generator engine
/// @tparam RealType the floating point type of the random numbers, defaults to double
///
/// This class template is a concrete derived class template for runtime-polymorphic generators. It uses
/// a variate_generator producing uniform random numbers in the interval[0,1) to fill the buffer of the 
/// @ref alps::buffered_generator  base class. 
/// If the @c Engine is a reference type,
//// a reference to the random number engine passed to the constructor is used. Otherwise a copy of the enigine is used.

template <class Engine, class RealType=double> 
class basic_buffered_uniform_01 
 : public buffered_uniform_01<RealType>
{
public:

  /// the type of random numbers
  typedef RealType result_type;
  
  /// the type of random number generator engine
  typedef Engine engine_type;
  
  typedef boost::uniform_real<RealType> distribution_type;
  
  typedef boost::variate_generator<engine_type,distribution_type> generator_type;

  
  /// constructs a default-seeded generator with a buffer of the size given as argument, 
  /// and uses a default-generated random number generator.
  /// @param buffer_size the size of the buffer

  basic_buffered_uniform_01(std::size_t buffer_size=ALPS_BUFFERED_GENERATOR_BUFFER_SIZE) 
   : buffered_uniform_01<RealType>(buffer_size)
   , generator_(generator_type(engine_type(),distribution_type()))
  {}

  /// constructs a generator from the given engine
  /// @param engine the engine used to generate values
  /// @param buffer_size the size of the buffer
  ///
  /// If a reference type is specifed as @c Engine type, a reference to the
  /// @c engine is stored and used, otherweise the engine is copied.
  basic_buffered_uniform_01(engine_type engine, std::size_t buffer_size=ALPS_BUFFERED_GENERATOR_BUFFER_SIZE) 
   : buffered_uniform_01<RealType>(buffer_size)
   , generator_(generator_type(engine,distribution_type()))
  {}

private:
  typedef typename buffered_generator<result_type>::buffer_type buffer_type;
  
  /// fills the buffer using the generator
  void fill_buffer(buffer_type& buffer)
  {
    for (typename buffer_type::iterator it=buffer.begin();it!=buffer.end();++it)
      *it=generator_();
  }

  generator_type generator_;
};


} // end namespace alps

#endif // ALPS_RANDOM_BUFFERED_UNIFORM_01_HPP
