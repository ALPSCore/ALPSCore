/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_RANDOM_H
#define ALPS_RANDOM_H

#include <alps/config.h>
#include <alps/random/pseudo_des.h>

#include <boost/random.hpp>
#include <boost/utility.hpp>
#include <boost/throw_exception.hpp>
#include <boost/generator_iterator.hpp>

#include <iostream>

namespace alps {

template <class RNG>
void seed_with_sequence(RNG& rng, uint32_t seed)
{
  pseudo_des start(seed);
  pseudo_des end(seed);
  start(); // make start!=end
  typedef boost::generator_iterator_generator<pseudo_des>::type iterator_type;
  iterator_type start_it(boost::make_generator_iterator(start));
  iterator_type end_it(boost::make_generator_iterator(end));
  rng.seed(start_it,end_it);
}

class BufferedRandomNumberGeneratorBase
{
public:
  typedef double result_type;
  typedef std::vector<result_type> buffer_type;
  
  BOOST_STATIC_CONSTANT(bool, has_fixed_range = false);

  BufferedRandomNumberGeneratorBase(std::size_t b=10240) 
   : buf_(b), ptr_(buf_.end()) {}

  BufferedRandomNumberGeneratorBase(const BufferedRandomNumberGeneratorBase& gen)
  {
    buf_ = gen.buf_;
    ptr_ = buf_.begin()+(gen.ptr_-const_cast<buffer_type&>(gen.buf_).begin());
  }

  virtual ~BufferedRandomNumberGeneratorBase() {}

  result_type operator()() {
    if(ptr_==buf_.end()) {
      fill_buffer();
      ptr_=buf_.begin();
    }
    return *ptr_++;
  }
  virtual void seed(uint32_t) = 0;
  virtual void seed() =0;
  virtual void write(std::ostream&) const =0;
  virtual void read(std::istream&)=0;

  result_type min() const { return result_type(0); }
  result_type max() const { return result_type(1); }

protected:
  std::vector<result_type> buf_;
  std::vector<result_type>::iterator ptr_;
private:
  virtual void fill_buffer() = 0;
};


template <class RNG> class BufferedRandomNumberGenerator
 : public BufferedRandomNumberGeneratorBase
{
public:
  BufferedRandomNumberGenerator() : rng_(), gen_(rng_,boost::uniform_real<>()) {}
  BufferedRandomNumberGenerator(RNG rng) : rng_(rng), gen_(rng_,boost::uniform_real<>()) {}

  void fill_buffer();
  template <class IT>
  void seed(IT start, IT end) { rng_.seed(start,end);}
  void seed(uint32_t);
  void seed();
  virtual void write(std::ostream&) const;
  virtual void read(std::istream&);
protected:
  RNG rng_;
  boost::variate_generator<RNG&,boost::uniform_real<> > gen_;
};



template <class RNG>
void BufferedRandomNumberGenerator<RNG>::seed(uint32_t s)
{
  seed_with_sequence(rng_,s);
}

template <class RNG>
void BufferedRandomNumberGenerator<RNG>::seed()
{
  rng_.seed();
}

template <class RNG>
void BufferedRandomNumberGenerator<RNG>::read(std::istream& is)
{
  is >> rng_;
}

template <class RNG>
void BufferedRandomNumberGenerator<RNG>::write(std::ostream& os) const
{
  os << rng_;
}

template <class RNG>
void BufferedRandomNumberGenerator<RNG>::fill_buffer()
{
//  std::generate(buf_.begin(),buf_.end(),gen_);
  std::vector<result_type>::iterator xx = buf_.begin();
  while (xx != buf_.end())
  {
    *xx = gen_();
    ++xx;
  }
}

} // end namespace

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& os, const BufferedRandomNumberGeneratorBase& r) {
  r.write(os);
  return os;
}

inline std::istream& operator>>(std::istream& is, BufferedRandomNumberGeneratorBase& r) {
  r.read(is);
  return is;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_RANDOM_H
