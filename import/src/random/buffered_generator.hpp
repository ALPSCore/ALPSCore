/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#ifndef ALPS_RANDOM_BUFFERED_GENERATOR_HPP
#define ALPS_RANDOM_BUFFERED_GENERATOR_HPP

#include <boost/assert.hpp>
#include <boost/ref.hpp>
#include <vector>
#include <algorithm>

/// @file This file declares and implements a polymorphic buffered generator
namespace alps {

#ifndef ALPS_BUFFERED_GENERATOR_BUFFER_SIZE

/// @brief the default bufer size
///
/// The default buffer size can be changed by defining the macro 
/// ALPS_BUFFERED_GENERATOR_BUFFER_SIZE
#define ALPS_BUFFERED_GENERATOR_BUFFER_SIZE 8192

#endif

/// @brief abstract base class of a runtime-polymorphic buffered_generator 
///
/// @tparam ResultType the type of values generated
/// 
/// This class template is an abstract base class for runtime-polymorphic generators.  
/// In order to mask the abstraction penalty of a virtual @c operator(), the buffered_generator
/// does not produce single numbers at each call, but instead a
/// large buffer is filled in a virtual function call, and then
/// numbers from this buffer used when @c operator() is called.


template <class ResultType>
class buffered_generator
{
public:
  //// the type of values generated
  typedef ResultType result_type;

  /// the data type of the buffer used
  typedef std::vector<result_type> buffer_type;
  
  /// @brief the constructor of the buffered generator
  ///
  /// @param buffer_size the size of the buffer
  ///
  /// Constructs a @c buffered_generator with a buffer of the size given as argument.

  buffered_generator(std::size_t buffer_size=ALPS_BUFFERED_GENERATOR_BUFFER_SIZE) 
   : buffer_(buffer_size)
   , ptr_(buffer_.end()) 
  {
    BOOST_ASSERT(buffer_size!=0);
  }

  /// the copy constructor copies the buffer
  buffered_generator(const buffered_generator& gen)
   : buffer_(gen.buffer_)
   , ptr_(buffer_.begin()+(gen.ptr_-gen.buffer_.begin()))
  {}

  /// trivial virtual destructor
  virtual ~buffered_generator() {}

  /// non-trivial the assignment
  buffered_generator& operator=(const buffered_generator& gen)
  {
    buffer_ = gen.buffer_;
    ptr_ = buffer_.begin()+(gen.ptr_-gen.buffer_.begin());
    return *this;
  }

  /// \brief returns the next generated value 
  ///
  /// values are taken from the buffer, which is refilled by a call
  /// to fill_buffer when all values have been used up
  result_type operator()() {
    if(ptr_==buffer_.end()) {
      fill_buffer(buffer_);
      ptr_=buffer_.begin();
    }
    BOOST_ASSERT(ptr_!=buffer_.end());
    return *ptr_++;
  }
  
  /// discards all elements in the buffers
  /// and forces a new call to fill_buffer when the next value is requested
  void reset()
  {
    ptr_=buffer_.end();
  }

//protected:
//  /// read access to the buffer for derived classes
//  buffer_type const& buffer() const
//  {
//    return buffer_;
//  }
  
private:
  /// \brief pure virtual function to fill the buffer
  /// 
  /// A pure virtual function to fill the buffer. It needs to be implemented by the concrete derived classes
  virtual void fill_buffer(buffer_type&) = 0;

  buffer_type buffer_;
  typename buffer_type::const_iterator ptr_;
};


/// @brief a concrete implementation of a buffered generator
/// 
/// @tparam Generator the type of the generator used to fill the buffer
/// @tparam ResultType the type of values generated
///
/// This class template is a concrete derived class template for runtime-polymorphic generators. It uses
/// the generator provided as template argument to fill the buffer of the @ref buffered_generator base class. 
/// If the @c Generator is a reference type, a reference to the generator passed to the constructor is used. 
/// Otherwise a copy of the generator is used.

template <class Generator, class ResultType> 
class basic_buffered_generator : public buffered_generator<ResultType>
{
  typedef buffered_generator<ResultType> base_type;
public:
  /// the date type of the generator used to fill the buffer
  typedef Generator generator_type;
  
  /// the type of values generated
  typedef typename buffered_generator<ResultType>::result_type result_type;
  
  /// constructs a buffer of the size given as argument, and a default-generated Generator.
  /// @param buffer_size the size of the buffer

  basic_buffered_generator(std::size_t buffer_size=ALPS_BUFFERED_GENERATOR_BUFFER_SIZE) 
   : base_type(buffer_size)
   {}
  
  /// constructs a buffered generator from the given argument
  /// @param gen the generator used to generate values
  /// @param buffer_size the size of the buffer
  ///
  /// If a reference type is specifed as \c Generator type, a reference to the generator
  /// @c gen is stored and used, otherweise the generator is copied.
  basic_buffered_generator(generator_type gen, std::size_t buffer_size=ALPS_BUFFERED_GENERATOR_BUFFER_SIZE) 
   : base_type(buffer_size)
   , generator_(gen) 
  {}

private:
  typedef typename buffered_generator<ResultType>::buffer_type buffer_type;
  
  /// fills the buffer using the generator
  void fill_buffer(buffer_type& buffer)
  {
    for (typename buffer_type::iterator it=buffer.begin();it!=buffer.end();++it)
      *it=generator_();
  }
  
  generator_type generator_;
};


} // end namespace alps


#endif // ALPS_RANDOM_BUFFERED_GENERATOR_HPP
