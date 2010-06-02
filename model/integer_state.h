/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2004 by Matthias Troyer <troyer@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_MODEL_INTEGER_STATE_H
#define ALPS_MODEL_INTEGER_STATE_H

#include <boost/integer/static_log2.hpp>

namespace alps {

template <class I, int N=1> class integer_state;

template <class I, int N>
class integer_state {
public:
  BOOST_STATIC_CONSTANT(int, bits = boost::static_log2<N>::value+1);
  BOOST_STATIC_CONSTANT(int, mask = (1<<bits)-1);
  typedef I representation_type;
  
  class reference {
  public:
    reference(I& s, int i) : state_(s), shift_(i*bits) {}
    operator int() const { return (state_ >> shift_) & mask;}
    template <class T>
    reference& operator=(T x)
    {
      state_ &= ~(mask<<shift_);
      state_ |= ((mask & x)<<shift_);
      return *this;
    }
  private:
    I& state_;
    std::size_t shift_;
  };
  
  integer_state(representation_type x=0) : state_(x) {}
  
  template <class J>
  integer_state(const std::vector<J>& x) : state_(0)
  { 
    for (int i=0;i<x.size();++i)  
      state_ |=(x[i]<<(i*bits));
  }
  int operator[](int i) const { return (state_>>i)&mask;}
  reference operator[](int i) { return reference(state_,i);}
  operator representation_type() const { return state_;}
  representation_type state() const { return state_;}
private:
  representation_type state_;
};

template <class I>
class integer_state<I,1> {
public:
  typedef I representation_type;
  
  class reference {
  public:
    reference(I& s, int i) : state_(s), mask_(1<<i) {}
    operator int() const { return (state_&mask_ ? 1 : 0);}
    template <class T>
    reference& operator=(T x)
    {
      if (x)
        state_|=mask_;
      else
        state_&=~mask_;
      return *this;
    }
  private:
    I& state_;
    I mask_;
  };
  
  integer_state(representation_type x=0) : state_(x) {}
  
  template <class J>
  integer_state(const std::vector<J>& x) : state_(0)
  { 
    for (int i=0;i<x.size();++i)  
      if(x[i])
        state_ |=(1<<i);
  }
  int operator[](int i) const { return (state_>>i)&1;}
  reference operator[](int i) { return reference(state_,i);}
  operator representation_type() const { return state_;}
  representation_type state() const { return state_;}
private:
  representation_type state_;
};

template <class I, int N>
bool operator == (integer_state<I,N> x, integer_state<I,N> y)
{ return x.state() == y.state(); }

template <class I, int N>
bool operator < (integer_state<I,N> x, integer_state<I,N> y)
{ return x.state() < y.state(); }

} // namespace alps

#endif
