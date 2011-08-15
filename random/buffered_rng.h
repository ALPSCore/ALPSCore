/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Mario Ruetti <mruetti@gmx.net>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id$ */

/// \file buffered_rng.h
/// \brief contains an efficient buffered implementation of a runtime-polymorphic random number generator

#ifndef ALPS_RANDOM_H
#define ALPS_RANDOM_H

#include <cmath>
#include <alps/config.h>
#include <alps/random/pseudo_des.h>
#include <alps/random/seed.h>
#include <alps/random/mersenne_twister.hpp>

#include <boost/integer_traits.hpp>
#include <boost/utility.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/detail/workaround.hpp>

#include <iostream>
#include <vector>

namespace alps {

/// \brief abstract base class of a runtime-polymorphic random number generator
///
/// In order to mask the abstraction penalty, the derived generators
/// do not produce single random numbers at each call, but instead a
/// large buffer is filled in a virtual function call, and then
/// numbers from this buffer used when operator() is called.
class buffered_rng_base
{
public:
  /// we create random numbers of type uint32_t
  typedef uint32_t result_type;
  typedef std::vector<result_type> buffer_type;

  BOOST_STATIC_CONSTANT(bool, has_fixed_range = false);

  /// \brief the constructor
  /// \param b the size of the buffer
  buffered_rng_base(std::size_t b=10240)
   : buf_(b), ptr_(buf_.end()) {}

  buffered_rng_base(const buffered_rng_base& gen)
  {
    buf_ = gen.buf_;
    ptr_ = buf_.begin()+(gen.ptr_-const_cast<buffer_type&>(gen.buf_).begin());
  }

  virtual ~buffered_rng_base() {}

  /// \brief returns the next random number
  ///
  /// numbers are taken from the buffer, which is refilled by a call
  /// to fill_buffer when it gets empty
  result_type operator()() {
    if(ptr_==buf_.end()) {
      fill_buffer();
      ptr_=buf_.begin();
    }
    return *ptr_++;
  }

  template <class OutputIterator>
  OutputIterator generate_n(std::size_t n, OutputIterator it);

  /// seed with an unsigned integer
  virtual void seed(uint32_t) = 0;
  /// seed with the default value
  virtual void seed() =0;
  /// seed with the pseudo_des generator
  virtual void seed(pseudo_des& inigen) = 0;
  /// write the state to a std::ostream
  virtual void write(std::ostream&) const =0;
  /// read the state from a std::istream
  virtual void read(std::istream&)=0;

  /// write the full state (including buffer) to a std::ostream
  virtual void write_all(std::ostream& os) const = 0;
  /// read the full state (including buffer) from a std::istream
  virtual void read_all(std::istream&) = 0;

  virtual result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const = 0;
  virtual result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const = 0;

protected:
  std::vector<result_type> buf_;
  std::vector<result_type>::iterator ptr_;

private:
  /// refills the buffer
  virtual void fill_buffer() = 0;
};

/// a concrete implementation of a buffered random number generator
/// \param RNG the type of random number generator
template <class RNG> class buffered_rng : public buffered_rng_base
{
private:
  BOOST_STATIC_ASSERT( (::boost::is_same<typename RNG::result_type, uint32_t>::value) );

public:
  /// constructs a default-seeded generator
  buffered_rng() : rng_() {}
  /// constructs a generator by copying the argument
  /// \param rng generator to be copied
  buffered_rng(RNG rng) : rng_(rng) {}

  template <class IT>
  void seed(IT start, IT end) { rng_.seed(start, end); }
  /// seed from an integer using seed_with_sequence
  /// \sa seed_with_sequence()
  void seed(uint32_t s) { seed_with_sequence(rng_,s); }
  void seed();
  /// seed with the pseudo_des generator
  void seed(pseudo_des& inigen) { seed_with_generator(rng_, inigen); }

  result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return rng_.min BOOST_PREVENT_MACRO_SUBSTITUTION (); }
  result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return rng_.max BOOST_PREVENT_MACRO_SUBSTITUTION (); }

  virtual void write(std::ostream&) const;
  virtual void read(std::istream&);
  virtual void write_all(std::ostream&) const;
  virtual void read_all(std::istream&);

protected:
  void fill_buffer();
  RNG rng_;
};

template <class RNG>
void buffered_rng<RNG>::seed()
{
  rng_.seed();
}

template <class RNG>
void buffered_rng<RNG>::read(std::istream& is)
{
  is >> rng_;
}

template <class RNG>
void buffered_rng<RNG>::write(std::ostream& os) const
{
  os << rng_;
}

template <class RNG>
void buffered_rng<RNG>::read_all(std::istream& is)
{
  int32_t n;
  is >> n;
  ptr_ = buf_.end() - n;
  for (std::vector<result_type>::iterator itr = ptr_; itr != buf_.end(); ++itr)
    is >> *itr;
  is >> rng_;
}

template <class RNG>
void buffered_rng<RNG>::write_all(std::ostream& os) const
{
  int32_t n = buf_.end() - ptr_;
  os << n << ' ';
  for (std::vector<result_type>::iterator itr = ptr_; itr != buf_.end(); ++itr)
    os << *itr << ' ';
  os << rng_;
}

template <class RNG>
void buffered_rng<RNG>::fill_buffer()
{
  // std::generate(buf_.begin(),buf_.end(),gen_);
  std::vector<result_type>::iterator xx = buf_.begin();
  while (xx != buf_.end())
  {
    *xx = rng_();
    ++xx;
  }
}

} // end namespace

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// writes the state of the generator to a std::ostream
/// \sa buffered_rng_base
inline std::ostream& operator<<(std::ostream& os, const buffered_rng_base& r) {
  r.write(os);
  return os;
}

/// reads the state of the generator from a std::istream
/// \sa buffered_rng_base
inline std::istream& operator>>(std::istream& is, buffered_rng_base& r) {
  r.read(is);
  return is;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_RANDOM_H
