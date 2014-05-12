/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_OSIRIS_DUMP_H
#define ALPS_OSIRIS_DUMP_H

#include <alps/config.h>
#ifdef ALPS_HAVE_STDARG_H
# include <stdarg.h>
#endif
#include <boost/smart_ptr.hpp>
#include <complex>
#include <iostream>
#include <map>
//#include <rpc/types.h>
#include <typeinfo>
#include <vector>
#include <string>

namespace alps {

class ALPS_DECL ODump {
public:
  ODump(uint32_t v = 0);
  virtual ~ODump() {}

  uint32_t version() const { return version_; }

/// INTERNAL ONLY
# define ALPS_DUMP_DO_TYPE(T) \
  virtual void write_simple(T x);
  ALPS_DUMP_DO_TYPE(bool)
  ALPS_DUMP_DO_TYPE(char)
  ALPS_DUMP_DO_TYPE(signed char)
  ALPS_DUMP_DO_TYPE(unsigned char)
  ALPS_DUMP_DO_TYPE(short)
  ALPS_DUMP_DO_TYPE(unsigned short)
  virtual void write_simple(int x) = 0;
  ALPS_DUMP_DO_TYPE(unsigned int)
  ALPS_DUMP_DO_TYPE(long)
  ALPS_DUMP_DO_TYPE(unsigned long)
# ifdef BOOST_HAS_LONG_LONG
  ALPS_DUMP_DO_TYPE(long long)
  ALPS_DUMP_DO_TYPE(unsigned long long)
# endif
  ALPS_DUMP_DO_TYPE(float)
  virtual void write_simple(double x) = 0;
  ALPS_DUMP_DO_TYPE(long double)
# undef ALPS_DUMP_DO_TYPE

  template <class T>
  void write_complex(const std::complex<T>& x)
  {
    write_simple(std::real(x));
    write_simple(std::imag(x));
  }

  // template<class T>
  // ODump& operator<<(const T& x) {x.save(*this); return *this; }
  // 
  // template<class T>
  // ODump& operator<<(const std::complex<T>& x) { write_complex(x); return *this; }

  template<class T>
  ODump& store(const T& x) {x.save(*this); return *this; }

  template<class T>
  ODump& store(const std::complex<T>& x) { write_complex(x); return *this; }

/// INTERNAL ONLY
# define ALPS_DUMP_DO_TYPE(T)                        \
  virtual void write_array(std::size_t n, const T * p); \
  ODump& operator<<( T x) { write_simple(x); return *this; }
  ALPS_DUMP_DO_TYPE(bool)
  ALPS_DUMP_DO_TYPE(char)
  ALPS_DUMP_DO_TYPE(signed char)
  ALPS_DUMP_DO_TYPE(unsigned char)
  ALPS_DUMP_DO_TYPE(short)
  ALPS_DUMP_DO_TYPE(unsigned short)
  ALPS_DUMP_DO_TYPE(int)
  ALPS_DUMP_DO_TYPE(unsigned int)
  ALPS_DUMP_DO_TYPE(long)
  ALPS_DUMP_DO_TYPE(unsigned long)
# ifdef BOOST_HAS_LONG_LONG
  ALPS_DUMP_DO_TYPE(long long)
  ALPS_DUMP_DO_TYPE(unsigned long long)
# endif
  ALPS_DUMP_DO_TYPE(float)
  ALPS_DUMP_DO_TYPE(double)
  ALPS_DUMP_DO_TYPE(long double)
# undef ALPS_DUMP_DO_TYPE

  template <class T>
  void write_array(std::size_t n, const std::complex<T>* p)
  { write_array(2 * n, reinterpret_cast<const T*>(p)); }

  virtual void write_string(std::size_t n, const char* s);
  virtual void write_string(const std::string&);

private:
  uint32_t version_;
};

template<class T>
ODump& operator<<(ODump& d, const T& x) {x.save(d); return d; }

template<class T>
ODump& operator<<(ODump& d, const std::complex<T>& x) { d.write_complex(x); return d; }

class ALPS_DECL IDump {
public:
  IDump(uint32_t v=0);
  virtual ~IDump() {}

  uint32_t version() const { return version_;}
  void set_version(uint32_t v) { version_=v;}

/// INTERNAL ONLY
# define ALPS_DUMP_DO_TYPE(T) \
  virtual void read_simple(T& x);

  ALPS_DUMP_DO_TYPE(bool)
  ALPS_DUMP_DO_TYPE(char)
  ALPS_DUMP_DO_TYPE(signed char)
  ALPS_DUMP_DO_TYPE(unsigned char)
  ALPS_DUMP_DO_TYPE(short)
  ALPS_DUMP_DO_TYPE(unsigned short)
  virtual void read_simple(int& x) = 0;
  ALPS_DUMP_DO_TYPE(unsigned int)
  ALPS_DUMP_DO_TYPE(long)
  ALPS_DUMP_DO_TYPE(unsigned long)
# ifdef BOOST_HAS_LONG_LONG
  ALPS_DUMP_DO_TYPE(long long)
  ALPS_DUMP_DO_TYPE(unsigned long long)
# endif
  ALPS_DUMP_DO_TYPE(float)
  virtual void read_simple(double& x) = 0;
  ALPS_DUMP_DO_TYPE(long double)
# undef ALPS_DUMP_DO_TYPE

  template <class T>
  void read_complex(std::complex<T>& x)
  {
    T re = get<T>();
    T im = get<T>();
    x = std::complex<T>(re,im);
  }

  // template<class T>
  // IDump& operator>>(T& x) { x.load(*this); return *this; }
  // 
  // template<class T>
  // IDump& operator>>(std::complex<T>& x) { read_complex(x); return *this; }

  template<class T>
  IDump& load(T& x) { x.load(*this); return *this; }

  template<class T>
  IDump& load(std::complex<T>& x) { read_complex(x); return *this; }

/// INTERNAL ONLY
# define ALPS_DUMP_DO_TYPE(T) \
  virtual void read_array(std::size_t n, T * p); \
  IDump& operator>>(T& x) { read_simple(x); return *this; } \
  operator T () { return get<T>(); }
  ALPS_DUMP_DO_TYPE(bool)
  ALPS_DUMP_DO_TYPE(char)
  ALPS_DUMP_DO_TYPE(signed char)
  ALPS_DUMP_DO_TYPE(unsigned char)
  ALPS_DUMP_DO_TYPE(short)
  ALPS_DUMP_DO_TYPE(unsigned short)
  ALPS_DUMP_DO_TYPE(int)
  ALPS_DUMP_DO_TYPE(unsigned int)
  ALPS_DUMP_DO_TYPE(long)
  ALPS_DUMP_DO_TYPE(unsigned long)
# ifdef BOOST_HAS_LONG_LONG
  ALPS_DUMP_DO_TYPE(long long)
  ALPS_DUMP_DO_TYPE(unsigned long long)
# endif
  ALPS_DUMP_DO_TYPE(float)
  ALPS_DUMP_DO_TYPE(double)
  ALPS_DUMP_DO_TYPE(long double)
# undef ALPS_DUMP_DO_TYPE

  template <class T>
  void read_array(std::size_t n, std::complex<T>* p)
  { read_array(2 * n, reinterpret_cast<T*>(p)); }

  virtual void read_string(std::size_t n, char* s);
  virtual void read_string(std::string&);

  template <class T>
  operator std::complex<T> ()
  {
    std::complex<T> x;
    read_simple(x);
    return x;
  }

  template <class T>
  inline T get()
  {
    T x; read_simple(x);
    return x;
  }

  // read the next boolean value from the dump and return its value.
  bool test() { return get<bool>(); }

private:
  uint32_t version_;
};

template<class T>
IDump& operator>>(IDump& d, T& x) { x.load(d); return d; }

template<class T>
IDump& operator>>(IDump& d, std::complex<T>& x) { d.read_complex(x); return d; }

} // end namespace

#include "dumparchive.h"

#endif // OSIRIS_DUMP_H
