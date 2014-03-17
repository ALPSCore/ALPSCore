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

#include <alps/osiris/xdrdump.h>
#include <alps/osiris/std/string.h>

#include <boost/throw_exception.hpp>
#include <boost/static_assert.hpp>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace alps {

#ifdef BOOST_HAS_LONG_LONG

namespace {

void static_assertion() {
  BOOST_STATIC_ASSERT(sizeof(long long) == 8);
}
    
} // end namespace

#endif

namespace detail {

bool xdr_bool(XDR *xdrs, bool *bp)
{
  if (xdrs->x_op == XDR_ENCODE) {
    bool_t b = *bp;
    return ::xdr_bool(xdrs, &b);
  } else if (xdrs->x_op == XDR_DECODE) {
    bool_t b;
    bool retval = ::xdr_bool(xdrs, &b);
    *bp=b;
    return retval;
  } else if (xdrs->x_op == XDR_FREE) {
    return true;
  }
  return false;
}

static bool xdr_s_char(XDR *xdrs, signed char *scp) 
{
  if (xdrs->x_op == XDR_ENCODE) {
    char c = *scp;
    return xdr_char(xdrs, &c);
  } else if (xdrs->x_op == XDR_DECODE) {
    char c;
    bool retval = ::xdr_char(xdrs, &c);
    *scp = c;
    return retval;
  } else if (xdrs->x_op == XDR_FREE) {
    return true;
  }
  return false;
}

bool xdr_hyper(XDR *xdrs, long long *llp)
{
#if defined( __LP64__ ) && defined(__APPLE__)
  int t1;
  unsigned int t2;
#else
  long t1;
  unsigned long t2;
#endif
  if (xdrs->x_op == XDR_ENCODE) {
    t1 = (long)((*llp) >> 32);
    t2 = (unsigned long)(*llp - (((long long) t1) << 32));
    return (::xdr_long(xdrs, &t1) && ::xdr_u_long(xdrs, &t2));
  } else if (xdrs->x_op == XDR_DECODE) {
    if (!::xdr_long(xdrs, &t1) || !::xdr_u_long(xdrs, &t2)) return false;
    *llp = ((long long) t1) << 32;
    *llp |= t2;
    return true;
  } else if (xdrs->x_op == XDR_FREE) {
    return true;
  }
  return false;
}

bool xdr_u_hyper(XDR *xdrs, unsigned long long *llp)
{
#if defined(__LP64__) && defined(__APPLE__)
  unsigned int t1;
  unsigned int t2;
#else
  unsigned long t1;
  unsigned long t2;
#endif
  if (xdrs->x_op == XDR_ENCODE) {
    t1 = (unsigned long)((*llp) >> 32);
    t2 = (unsigned long)(*llp - (((unsigned long long) t1) << 32));
    return (::xdr_u_long(xdrs, &t1) && ::xdr_u_long(xdrs, &t2));
  } else if (xdrs->x_op == XDR_DECODE) {
    if (!::xdr_u_long(xdrs, &t1) || !::xdr_u_long(xdrs, &t2)) return false;
    *llp = ((unsigned long long) t1) << 32;
    *llp |= t2;
    return true;
  } else if (xdrs->x_op == XDR_FREE) {
    return true;
  }
  return false;
}

bool xdr_long_8(XDR *xdrs, long *lp)
{
  long long t;
  if (xdrs->x_op == XDR_ENCODE) {
    t = (long long)(*lp);
    return alps::detail::xdr_hyper(xdrs, &t);
  } else if (xdrs->x_op == XDR_DECODE) {
    if (!alps::detail::xdr_hyper(xdrs, &t)) return false;
    *lp = (long)t;
    return true;
  } else if (xdrs->x_op == XDR_FREE) {
    return true;
  }
  return false;
}

bool xdr_u_long_8(XDR *xdrs, unsigned long *lp)
{
  unsigned long long t;
  if (xdrs->x_op == XDR_ENCODE) {
    t = (unsigned long long)(*lp);
    return alps::detail::xdr_u_hyper(xdrs, &t);
  } else if (xdrs->x_op == XDR_DECODE) {
    if (!alps::detail::xdr_u_hyper(xdrs, &t)) return false;
    *lp = (unsigned long)t;
    return true;
  } else if (xdrs->x_op == XDR_FREE) {
    return true;
  }
  return false;
}

bool xdr_long_double(XDR *xdrs, long double *ldp) 
{
  if (xdrs->x_op == XDR_ENCODE) {
    double high = *ldp;
    double low  = (*ldp-high);
    return xdr_double(xdrs, &high) && xdr_double(xdrs, &low);
  } else if (xdrs->x_op == XDR_DECODE) {
    double high = 0.;
    double low  = 0.;
    bool retval = xdr_double(xdrs, &high) && xdr_double(xdrs, &low); 
    *ldp = low + high;
    return retval;
  } else if (xdrs->x_op == XDR_FREE) {
    return true;
  }
  return false;
}

template<class T, int N>
struct xdr_helper {};

#define ALPS_DUMP_DO_TYPE(T,X) \
  template<int N> struct xdr_helper<T, N> { \
    static bool xdr_do_type(XDR * xdrs, T * v) { return X (xdrs, v); } \
  };
#define ALPS_DUMP_DO_TYPE_N(T,N,X) \
  template<> struct xdr_helper<T, N> { \
    static bool xdr_do_type(XDR * xdrs, T * v) { return X (xdrs, v); } \
  };
ALPS_DUMP_DO_TYPE(bool, alps::detail::xdr_bool)
ALPS_DUMP_DO_TYPE(char, xdr_char)
ALPS_DUMP_DO_TYPE(signed char, xdr_s_char)
ALPS_DUMP_DO_TYPE(unsigned char, xdr_u_char)
ALPS_DUMP_DO_TYPE(short, xdr_short)
ALPS_DUMP_DO_TYPE(unsigned short, xdr_u_short)
ALPS_DUMP_DO_TYPE(int, xdr_int)
ALPS_DUMP_DO_TYPE(unsigned int, xdr_u_int)
#if defined (__LP64__) && defined(__APPLE__)
ALPS_DUMP_DO_TYPE_N(int, 4, xdr_long)
ALPS_DUMP_DO_TYPE_N(unsigned int, 4, xdr_u_long)
#else
ALPS_DUMP_DO_TYPE_N(long, 4, xdr_long)
ALPS_DUMP_DO_TYPE_N(unsigned long, 4, xdr_u_long)
#endif
ALPS_DUMP_DO_TYPE_N(long, 8, xdr_long_8)
ALPS_DUMP_DO_TYPE_N(unsigned long, 8, xdr_u_long_8)
#ifdef BOOST_HAS_LONG_LONG
ALPS_DUMP_DO_TYPE(long long, alps::detail::xdr_hyper)
ALPS_DUMP_DO_TYPE(unsigned long long, alps::detail::xdr_u_hyper)
#endif
ALPS_DUMP_DO_TYPE(float, xdr_float)
ALPS_DUMP_DO_TYPE(double, xdr_double)
ALPS_DUMP_DO_TYPE(long double, xdr_long_double)
#undef ALPS_DUMP_DO_TYPE
#undef ALPS_DUMP_DO_TYPE_N

} // namespace detail

//-----------------------------------------------------------------------
// get and set the position in the stream
//-----------------------------------------------------------------------

uint32_t OXDRDump::getPosition() const
{
  return xdr_getpos((XDR*) &xdr_); // cast to non-const necessary
}

void OXDRDump::setPosition(uint32_t pos)
{
  if (!xdr_setpos(&xdr_,pos))
    boost::throw_exception(std::runtime_error("failed to reposition OXDRDump"));
}

#define ALPS_DUMP_DO_TYPE(T) \
void OXDRDump::write_simple(T x)  \
{ \
  if (!detail::xdr_helper<T, int(sizeof(T))>::xdr_do_type(&xdr_, const_cast<T*>(&x))) \
    boost::throw_exception(std::runtime_error("failed to write type "#T" to an OXDRDump"));\
} \
void OXDRDump::write_array(size_t n, const T* p)  \
{ \
  int l = n; \
  if (!xdr_vector(&xdr_, reinterpret_cast<char*>(const_cast<T*>(p)), l, int(sizeof(T)), (xdrproc_t) &detail::xdr_helper<T, int(sizeof(T))>::xdr_do_type)) \
    boost::throw_exception ( std::runtime_error("failed to write array of type "#T" to an OXDRDump")); \
} \
void IXDRDump::read_simple(T& x)\
{ \
  if (!detail::xdr_helper<T, int(sizeof(T))>::xdr_do_type(&xdr_, &x)) \
    boost::throw_exception(std::runtime_error("failed to read type "#T" from an IXDRDump")); \
} \
void IXDRDump::read_array(size_t n, T* p) \
{ \
  int l = n; \
  if (!xdr_vector(&xdr_, reinterpret_cast<char*>(p), l, int(sizeof(T)), (xdrproc_t) &detail::xdr_helper<T, int(sizeof(T))>::xdr_do_type)) \
    boost::throw_exception ( std::runtime_error("failed to read array of type "#T" from an IXDRDump")); \
}

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

void OXDRDump::write_string(size_t n,const char *p) 
{ 
  int l=n; 
  char* ncp = const_cast<char*>(p);
  if (!xdr_string(&xdr_,&ncp,l))
    boost::throw_exception ( std::runtime_error("failed to write a string to an OXDRDump"));
} 

void IXDRDump::read_string(size_t n, char *p) 
{ 
  int l=n; 
  if (!xdr_string(&xdr_,&p,l))
    boost::throw_exception ( std::runtime_error("failed to read a string from an IXDRDump"));
} 

//-----------------------------------------------------------------------
// get and set the position in the stream
//-----------------------------------------------------------------------

uint32_t IXDRDump::getPosition() const 
{
  return xdr_getpos((XDR*) &xdr_);
}

void IXDRDump::setPosition(uint32_t pos)
{
  if (!xdr_setpos(&xdr_,pos))
    boost::throw_exception( std::runtime_error("failed to reposition IXDRDump"));
}

//=======================================================================
// OXDRFileDump
// 
// implements a dump for writing into a file using the XDR format
//-----------------------------------------------------------------------

// reopen a file
void OXDRFileDump::open_file(const std::string& fn,bool append)
{
  file_ = std::fopen(fn.c_str(),(append ? "ab" : "wb"));
  if(file_)
      xdrstdio_create(&xdr_,file_,XDR_ENCODE);
  else  {
      // opening failed
      std::string text = "failed to open file \"";
      text += fn;
      text += "\" for writing";
      boost::throw_exception(std::runtime_error(text));
    }
}


// create a new dump file
OXDRFileDump::OXDRFileDump(const boost::filesystem::path& fn, bool append)
{
  open_file(fn.string(),append);
}

// destructor closes the stream and file
OXDRFileDump::~OXDRFileDump()
{
  xdr_destroy(&xdr_);
  if(file_)
    std::fclose(file_);
}

void OXDRFileDump::flush() 
{
  std::fflush(file_);
}


//=======================================================================
// IXDRFileDump
// 
// implements a dump for reading from a file using the XDR format
//-----------------------------------------------------------------------

// open a dump file
IXDRFileDump::IXDRFileDump(const boost::filesystem::path& p)
{
  open_file(p.string());
}

// open a file for reading at a specified position
void IXDRFileDump::open_file(const std::string& fn)
{
  valid_ = true;
  file_ = std::fopen(fn.c_str(),"rb");

  if(file_) // open succeeded
    xdrstdio_create(&xdr_,file_,XDR_DECODE);
  else   {
      // open failed
      std::string text = "failed to open file ";
      text += fn;
      text += " for reading";
      valid_=false;
#ifndef BOOST_NO_EXCEPTIONS
      boost::throw_exception (std::runtime_error(text));
#else
      std::cerr << "Osiris error: " << text << "\n";
#endif
    }
}


// destructor closes XDR stream and the file
IXDRFileDump::~IXDRFileDump()
{
  if(valid_) {
      xdr_destroy(&xdr_);
      if(file_)
        std::fclose(file_);
    }
}

} // namespace alps
