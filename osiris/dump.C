/***************************************************************************
* PALM++/osiris library
*
* osiris/dump.C      dumps for object serialization
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

#include <alps/osiris/dump.h>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {

ODump::ODump(uint32_t v) : version_(v), highestNumber_(0) {}

#define ALPS_DUMP_DO_TYPE(A, B) \
void ODump::write_simple(A x) \
{ write_simple(static_cast<B>(x)); }
ALPS_DUMP_DO_TYPE(bool, int32_t)
ALPS_DUMP_DO_TYPE(int8_t, int16_t)
ALPS_DUMP_DO_TYPE(uint8_t, int8_t)
ALPS_DUMP_DO_TYPE(int16_t, int32_t)
ALPS_DUMP_DO_TYPE(uint16_t, int16_t)
ALPS_DUMP_DO_TYPE(uint32_t, int32_t)
# ifndef BOOST_NO_INT64_T
ALPS_DUMP_DO_TYPE(int64_t, int32_t)
ALPS_DUMP_DO_TYPE(uint64_t, int64_t)
# endif
ALPS_DUMP_DO_TYPE(float, double)
ALPS_DUMP_DO_TYPE(long double, double)
#undef ALPS_DUMP_DO_TYPE

//-----------------------------------------------------------------------
// write_array
// 
// simply writes each element.
// should be implemented in an optimized way by the derived classes
//-----------------------------------------------------------------------

# define ALPS_DUMP_DO_TYPE(T) \
void ODump::write_array(std::size_t n, const T * p) \
{ for (std::size_t i = 0; i < n; ++i) write_simple(p[i]); }
ALPS_DUMP_DO_TYPE(bool)
ALPS_DUMP_DO_TYPE(int8_t)
ALPS_DUMP_DO_TYPE(uint8_t)
ALPS_DUMP_DO_TYPE(int16_t)
ALPS_DUMP_DO_TYPE(uint16_t)
ALPS_DUMP_DO_TYPE(int32_t)
ALPS_DUMP_DO_TYPE(uint32_t)
# ifndef BOOST_NO_INT64_T
ALPS_DUMP_DO_TYPE(int64_t)
ALPS_DUMP_DO_TYPE(uint64_t)
# endif
ALPS_DUMP_DO_TYPE(float)
ALPS_DUMP_DO_TYPE(double)
ALPS_DUMP_DO_TYPE(long double)
#undef ALPS_DUMP_DO_TYPE

void ODump::write_string(std::size_t n, const char* p) 
{
  for (std::size_t i=0;i<n;i++) write_simple(uint8_t(p[i]));
}

// pointers are stored in an assocatiove array and assigned numbers
void ODump::registerObjectAddress(void* p) 
{
  numberOfPointer_[p]=++highestNumber_;
}

// instead of a pointer its number is writen.
void ODump::writePointer(void* p)
{ 
  uint32_t n = numberOfPointer_[p];
  if (n == 0) {
    boost::throw_exception(std::runtime_error("pointer not registered"));
  } else {
    *this << n;
  }
}

IDump::IDump(uint32_t v) : version_(v) {}

//-----------------------------------------------------------------------
// operator >> for simple data types
//-----------------------------------------------------------------------

#define ALPS_DUMP_DO_TYPE(A, B) \
void IDump::read_simple(A& x) \
{ x = get<B>(); }
ALPS_DUMP_DO_TYPE(bool, int32_t)
ALPS_DUMP_DO_TYPE(int8_t, int16_t)
ALPS_DUMP_DO_TYPE(uint8_t, int8_t)
ALPS_DUMP_DO_TYPE(int16_t, int32_t)
ALPS_DUMP_DO_TYPE(uint16_t, int16_t)
ALPS_DUMP_DO_TYPE(uint32_t, int32_t)
# ifndef BOOST_NO_INT64_T
ALPS_DUMP_DO_TYPE(int64_t, int32_t)
ALPS_DUMP_DO_TYPE(uint64_t, int64_t)
# endif
ALPS_DUMP_DO_TYPE(float, double)
ALPS_DUMP_DO_TYPE(long double, double)
#undef ALPS_DUMP_DO_TYPE

//-----------------------------------------------------------------------
// read_array
// 
// simply reads each element.
// should be implemented in an optimized way by the derived classes
//-----------------------------------------------------------------------

# define ALPS_DUMP_DO_TYPE(T) \
void IDump::read_array(std::size_t n, T * p) \
{ for (std::size_t i = 0; i < n; ++i) read_simple(p[i]); }
ALPS_DUMP_DO_TYPE(bool)
ALPS_DUMP_DO_TYPE(int8_t)
ALPS_DUMP_DO_TYPE(uint8_t)
ALPS_DUMP_DO_TYPE(int16_t)
ALPS_DUMP_DO_TYPE(uint16_t)
ALPS_DUMP_DO_TYPE(int32_t)
ALPS_DUMP_DO_TYPE(uint32_t)
# ifndef BOOST_NO_INT64_T
ALPS_DUMP_DO_TYPE(int64_t)
ALPS_DUMP_DO_TYPE(uint64_t)
# endif
ALPS_DUMP_DO_TYPE(float)
ALPS_DUMP_DO_TYPE(double)
ALPS_DUMP_DO_TYPE(long double)
#undef ALPS_DUMP_DO_TYPE

void IDump::read_string(std::size_t n, char* p) 
{
  uint8_t c;
  for (std::size_t i = 0; i < n; ++i) {
    read_simple(c);
    p[i] = c;
  }
}

void IDump::registerObjectAddress(void* p)
{ pointerVector_.push_back(p); }

void* IDump::readPointer()
{
  int32_t n(*this);
  if (n >= pointerVector_.size())
    boost::throw_exception(std::runtime_error("pointer not registered"));
  return pointerVector_[n];
}

} // end namespace
