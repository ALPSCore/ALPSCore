/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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
