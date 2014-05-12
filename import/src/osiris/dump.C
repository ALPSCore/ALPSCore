/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/osiris/dump.h>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {

ODump::ODump(uint32_t v) : version_(v) {}

#define ALPS_DUMP_DO_TYPE(A, B) \
void ODump::write_simple(A x) \
{ write_simple(static_cast<B>(x)); }
ALPS_DUMP_DO_TYPE(bool, int)
ALPS_DUMP_DO_TYPE(char, short)
ALPS_DUMP_DO_TYPE(signed char, char)
ALPS_DUMP_DO_TYPE(unsigned char, char)
ALPS_DUMP_DO_TYPE(short, int)
ALPS_DUMP_DO_TYPE(unsigned short, short)
ALPS_DUMP_DO_TYPE(unsigned int, int)
ALPS_DUMP_DO_TYPE(long, int)
ALPS_DUMP_DO_TYPE(unsigned long, long)
#ifdef BOOST_HAS_LONG_LONG
ALPS_DUMP_DO_TYPE(long long, long)
ALPS_DUMP_DO_TYPE(unsigned long long, long long)
#endif
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
ALPS_DUMP_DO_TYPE(char)
ALPS_DUMP_DO_TYPE(signed char)
ALPS_DUMP_DO_TYPE(unsigned char)
ALPS_DUMP_DO_TYPE(short)
ALPS_DUMP_DO_TYPE(unsigned short)
ALPS_DUMP_DO_TYPE(int)
ALPS_DUMP_DO_TYPE(unsigned int)
ALPS_DUMP_DO_TYPE(long)
ALPS_DUMP_DO_TYPE(unsigned long)
#ifdef BOOST_HAS_LONG_LONG
ALPS_DUMP_DO_TYPE(long long)
ALPS_DUMP_DO_TYPE(unsigned long long)
#endif
ALPS_DUMP_DO_TYPE(float)
ALPS_DUMP_DO_TYPE(double)
ALPS_DUMP_DO_TYPE(long double)
#undef ALPS_DUMP_DO_TYPE

void ODump::write_string(std::size_t n, const char* p) 
{
  for (std::size_t i=0;i<n;i++) write_simple(uint8_t(p[i]));
}

void ODump::write_string(const std::string& s) 
{
  (*this) << uint32_t(s.size());
  if(s.size())
    write_string(s.size()+1,s.c_str());
}

IDump::IDump(uint32_t v) : version_(v) {}

//-----------------------------------------------------------------------
// operator >> for simple data types
//-----------------------------------------------------------------------

#define ALPS_DUMP_DO_TYPE(A, B) \
void IDump::read_simple(A& x) \
{ x = get<B>(); }
ALPS_DUMP_DO_TYPE(bool, int32_t)
ALPS_DUMP_DO_TYPE(char, short)
ALPS_DUMP_DO_TYPE(signed char, char)
ALPS_DUMP_DO_TYPE(unsigned char, char)
ALPS_DUMP_DO_TYPE(short, int)
ALPS_DUMP_DO_TYPE(unsigned short, short)
ALPS_DUMP_DO_TYPE(unsigned int, int)
ALPS_DUMP_DO_TYPE(long, int)
ALPS_DUMP_DO_TYPE(unsigned long, long)
#ifdef BOOST_HAS_LONG_LONG
ALPS_DUMP_DO_TYPE(long long, long)
ALPS_DUMP_DO_TYPE(unsigned long long, long long)
#endif
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

void IDump::read_string(std::size_t n, char* p) 
{
  uint8_t c;
  for (std::size_t i = 0; i < n; ++i) {
    read_simple(c);
    p[i] = c;
  }
}

void IDump::read_string(std::string& s) 
{
  uint32_t sz(*this);
  if(sz) {
    char* t = new char[sz+1];
    read_string(sz+1,t);
    if(t[sz]!=char(0))
      boost::throw_exception(std::runtime_error("string on dump not terminating with '\\0'"));
        s=t;
    delete[] t;
    if(s.length()!=sz)
      boost::throw_exception(std::runtime_error("string on dump has incorrect length"));
  } 
  else
    s="";
}

} // end namespace
