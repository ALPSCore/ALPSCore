/***************************************************************************
* PALM++/osiris library
*
* osiris/buffer.h      dumps for object serialization
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#ifndef OSIRIS_BUFFER_H
#define OSIRIS_BUFFER_H

#include <iostream>
// #include <palm/config.h>
#include <alps/osiris/dump.h>
#include <complex>

namespace alps {
namespace detail {

class Buffer;

/** a simple Buffer class. values can be written into it or read from it. */
class Buffer : public std::vector<uint8_t>
{
  public:
  /** create a buffer. 
      @param n the inital memory size allocated for the buffer.*/
  Buffer(size_type n=0)
    : std::vector<uint8_t>(n),
      read_pos(0),
      write_pos(0) {}
  /// deserialize the Buffer
  Buffer(IDump&);
  virtual ~Buffer() {};
  /// serialize the Buffer
  virtual void save(ODump&) const;
  /// deserialize the Buffer
  virtual void load(IDump&);

 /// erase the Buffer
 void clear() {(*this)=Buffer(0);}

  /** get a pointer to the Buffer. 
      This pointer might be invalidated by writing to the Buffer.
  */
  operator uint8_t* () { return (size() ? &(this->operator[](0)) : 0); }

//-----------------------------------------------------------------------
// READING AND WRITING
//-----------------------------------------------------------------------
 
  // write basic data types and arrays of them

#ifndef BOOST_NO_MEMBER_TEMPLATES 
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
  void read(T* p,size_type n=1) 
  {
    read_buffer(p, n*sizeof(T));
  }

  template <class T>
  void read(T& x) 
  {
    read_buffer(&x, sizeof(T));
  }
#else
#define IMPLEMENT_READ_WRITE(T) \
void read(T* p,size_type n=1) { read_buffer(p, n*sizeof(T)); } \
void read(T& x)  { read_buffer(&x, sizeof(T));} \
void write(const T* p,size_type n) { write_buffer(p, n*sizeof(T));} \
void write(const T x) { write_buffer(&x, sizeof(T));}

IMPLEMENT_READ_WRITE( bool x) { operator=(x);}
IMPLEMENT_READ_WRITE( int8_t x) { operator=(x);}
IMPLEMENT_READ_WRITE( int16_t x) { operator=(x);}
IMPLEMENT_READ_WRITE( int32_t x) { operator=(x);}
IMPLEMENT_READ_WRITE( int64_t x) { operator=(x);}
IMPLEMENT_READ_WRITE( uint8_t x) { operator=(x);}
IMPLEMENT_READ_WRITE( uint16_t x) { operator=(x);}
IMPLEMENT_READ_WRITE( uint32_t x) { operator=(x);}
IMPLEMENT_READ_WRITE( uint64_t x) { operator=(x);}
IMPLEMENT_READ_WRITE( float x) { operator=(x);}
IMPLEMENT_READ_WRITE( double x) { operator=(x);}
IMPLEMENT_READ_WRITE( long double x) { operator=(x);}
IMPLEMENT_READ_WRITE( std::complex<float> x) { operator=(x);}
IMPLEMENT_READ_WRITE( std::complex<double> x) { operator=(x);}
IMPLEMENT_READ_WRITE( std::complex<long double> x) { operator=(x);}
#endif

private:
  // the amount in multiples of which the Buffer grows
  static const size_type buffer_grow_steps=10240; 
  
  // the position at which reading will take place
  uint32_t read_pos; 
  // the position at which writing will take place
  uint32_t write_pos; 

//-----------------------------------------------------------------------
// READING AND WRITING
//-----------------------------------------------------------------------

  void write_buffer(const void*, size_type);
  void read_buffer(void*, size_type);
};

} // end namespace detail
} // end namespace alps

#endif // OSIRIS_BUFFER_H
