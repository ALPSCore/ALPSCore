/***************************************************************************
* PALM++/osiris library
*
* osiris/xdrdump.C      dumps for object serialization sing XDR
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/osiris/xdrdump.h>
#include <alps/osiris/std/string.h>

#include <boost/throw_exception.hpp>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace alps {

namespace detail {

static bool xdr_bool(XDR *xdr, bool *bp)
{
  if (xdr->x_op == XDR_ENCODE) {
    bool_t b = *bp;
    return ::xdr_bool(xdr, &b);
  } else if (xdr->x_op == XDR_DECODE) {
    bool_t b;
    bool retval = ::xdr_bool(xdr, &b);
    *bp=b;
    return retval;
  }
  return true;
}

static bool xdr_long_double(XDR *xdr, long double *ldp) 
{
  if (xdr->x_op == XDR_ENCODE) {
    double high = *ldp;
    double low  = (*ldp-high);
    return xdr_double(xdr, &high) && xdr_double(xdr, &low);
  } else if (xdr->x_op == XDR_DECODE) {
    double high;
    double low;
    bool retval = xdr_double(xdr, &high) && xdr_double(xdr, &low); 
    *ldp = low + high;
    return retval;
  }
  return true;
}

template<class T>
struct xdr_helper;

template<>
struct xdr_helper<char>
{
  static bool xdr_int8_t(XDR *xdr, char *v)
  { return xdr_char(xdr, v); };
};

template<>
struct xdr_helper<signed char>
{
  static bool xdr_int8_t(XDR *xdr, signed char *scp) 
  {
    if (xdr->x_op == XDR_ENCODE) {
      char c = *scp;
      return xdr_char(xdr, &c);
    } else if (xdr->x_op == XDR_DECODE) {
      char c;
      bool retval = xdr_char(xdr, &c);
      *scp = c;
      return retval;
    }
    return true;
  }
};

template<>
struct xdr_helper<int>
{
  static bool xdr_int32_t(XDR *xdr, int *v)
  { return xdr_int(xdr, v); };
};

template<>
struct xdr_helper<long>
{
  static bool xdr_int32_t(XDR *xdr, long *v)
  { return xdr_long(xdr, v); };
};

template<>
struct xdr_helper<unsigned int>
{
  static bool xdr_uint32_t(XDR *xdr, unsigned int *v)
  { return xdr_u_int(xdr, v); };
};

template<>
struct xdr_helper<unsigned long>
{
  static bool xdr_uint32_t(XDR *xdr, unsigned long *v)
  { return xdr_u_long(xdr, v); };
};

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

#define ALPS_DUMP_DO_TYPE(T,XDR) \
void OXDRDump::write_simple(T x)  \
{ \
  if (!XDR(&xdr_, const_cast<T*>(&x))) \
    boost::throw_exception(std::runtime_error("failed to write type "#T" to an OXDRDump"));\
} \
void OXDRDump::write_array(size_t n, const T* p)  \
{ \
  int l = n; \
  if (!xdr_vector(&xdr_, reinterpret_cast<char*>(const_cast<T*>(p)), l, sizeof(T), (xdrproc_t) &XDR)) \
    boost::throw_exception ( std::runtime_error("failed to write array of type "#T" to an OXDRDump")); \
} \
void IXDRDump::read_simple(T& x)\
{ \
  if (!XDR(&xdr_, &x)) \
    boost::throw_exception(std::runtime_error("failed to read type "#T" from an IXDRDump")); \
} \
void IXDRDump::read_array(size_t n, T* p) \
{ \
  int l = n; \
  if (!xdr_vector(&xdr_, reinterpret_cast<char*>(p), l, sizeof(T), (xdrproc_t) &XDR)) \
    boost::throw_exception ( std::runtime_error("failed to read array of type "#T" from an IXDRDump")); \
}

ALPS_DUMP_DO_TYPE(bool, detail::xdr_bool)
ALPS_DUMP_DO_TYPE(int8_t, detail::xdr_helper<int8_t>::xdr_int8_t)
ALPS_DUMP_DO_TYPE(uint8_t, xdr_u_char)
ALPS_DUMP_DO_TYPE(int16_t, xdr_short)
ALPS_DUMP_DO_TYPE(uint16_t, xdr_u_short)
ALPS_DUMP_DO_TYPE(int32_t, detail::xdr_helper<int32_t>::xdr_int32_t)
ALPS_DUMP_DO_TYPE(uint32_t, detail::xdr_helper<uint32_t>::xdr_uint32_t)
# ifndef BOOST_NO_INT64_T
ALPS_DUMP_DO_TYPE(int64_t, xdr_hyper)
ALPS_DUMP_DO_TYPE(uint64_t, xdr_u_hyper)
# endif
ALPS_DUMP_DO_TYPE(float, xdr_float)
ALPS_DUMP_DO_TYPE(double, xdr_double)
ALPS_DUMP_DO_TYPE(long double, detail::xdr_long_double)

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
// derived from OXDRDump
//
// implements a dump for writing into a file using the XDR format
//-----------------------------------------------------------------------

// reopen a file
void OXDRFileDump::open_file(const std::string& fn)
{
  file_ = std::fopen(fn.c_str(),"w");
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


// // create a new dump file
// OXDRFileDump::OXDRFileDump(const std::string& fn)
// {
//   open_file(fn);
// }

// create a new dump file
OXDRFileDump::OXDRFileDump(const boost::filesystem::path& fn)
{
  open_file(fn.native_file_string());
}

// destructor closes the stream and file
OXDRFileDump::~OXDRFileDump()
{
  xdr_destroy(&xdr_);
  if(file_)
    std::fclose(file_);
}


//=======================================================================
// IXDRFileDump
// 
// derived from IXDRDump
//
// implements a dump for reading from a file using the XDR format
//-----------------------------------------------------------------------

// // open a dump file
// IXDRFileDump::IXDRFileDump(const std::string& fn)
// {
//   open_file(fn);
// }

// open a dump file
IXDRFileDump::IXDRFileDump(const boost::filesystem::path& p)
{
  open_file(p.native_file_string());
}

// open a file for reading at a specified position
void IXDRFileDump::open_file(const std::string& fn)
{
  valid_ = true;
  file_ = std::fopen(fn.c_str(),"r");

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
