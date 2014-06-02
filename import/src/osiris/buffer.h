/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef OSIRIS_BUFFER_H
#define OSIRIS_BUFFER_H

#include <alps/config.h>
#include <vector>

namespace alps {
namespace detail {

class Buffer;

/** a simple Buffer class. values can be written into it or read from it. */
class Buffer : public std::vector<char>
{
  public:
  /** create a buffer. */
  Buffer()
    : std::vector<char>(),
      read_pos(0) {}

 /// erase the Buffer
 void clear() {(*this)=Buffer();}

  /** get a pointer to the Buffer. 
      This pointer might be invalidated by writing to the Buffer.
  */
  operator char* () { return (size() ? &(this->operator[](0)) : 0); }

  // write basic data types and arrays of them

template <class T>
  void write(const T* p,size_type n)
  {
    write_buffer(p, n*sizeof(T));
  }

  template <class T>
  void write(const T x) 
  {
    write_buffer(&x, sizeof(T));
  }

  // read basic data types and arrays of them
  
  template <class T>
  void read(T* p,size_type n=1) { read_buffer(p, n*sizeof(T));}

  template <class T>
  void read(T& x) { read_buffer(&x, sizeof(T)); }

private:
  // the position at which reading will take place
  uint32_t read_pos; 

  void write_buffer(const void*, size_type);
  void read_buffer(void*, size_type);
};

} // end namespace detail
} // end namespace alps

#endif // OSIRIS_BUFFER_H
