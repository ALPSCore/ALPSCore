/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
