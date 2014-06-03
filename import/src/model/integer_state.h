/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
