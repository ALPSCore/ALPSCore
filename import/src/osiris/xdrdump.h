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

#ifndef OSIRIS_XDRDUMP_H
#define OSIRIS_XDRDUMP_H

#include <alps/config.h>
#include <alps/osiris/dump.h>
#include <boost/filesystem/path.hpp>
#include <boost/cstdint.hpp>
#include <cstdio>
#include <string>
#include <stdio.h>

#ifdef ALPS_HAVE_RPC_XDR_H
#include <rpc/rpc.h>
#else
#include <alps/osiris/xdrcore.h>
#endif
// remove harmful 'enum_t' macro
// (which conflicts with boost/detail/scoped_enum_emulation.hpp in Boost 1.41.0)
#ifdef enum_t
# undef enum_t
#endif

#ifdef BOOST_NO_STDC_NAMESPACE
  namespace std {
    using ::FILE;
    using ::fopen;
    using ::fclose;
    using ::ftell;
  }
#endif

namespace alps {

/** The abstract base class for serializing an object
    using the XDR stream library to write the architecture
    indepedent XDR format. */

class ALPS_DECL OXDRDump : public ODump
{
public:
  OXDRDump () : ODump(0) {}
  virtual ~OXDRDump() {}

/// INTERNAL ONLY
# define ALPS_DUMP_DO_TYPE(T) \
  void write_simple(T x); \
  void write_array(std::size_t, const T *);
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

  // write a c-style string
  virtual void write_string(std::size_t, const char *);

protected:
  /// get the position in the XDR stream.
  uint32_t getPosition() const;
  /// set the position in the XDR stream.
  void setPosition(uint32_t pos);

  XDR xdr_; // the XDR stream
};


/** The abstract base class for deserializing an object
    using the XDR stream library to read the architecture
    indepedent XDR format. */

class ALPS_DECL IXDRDump : public IDump
{
public:
  IXDRDump() : IDump(0) {}
  virtual ~IXDRDump() {}

/// INTERNAL ONLY
# define ALPS_DUMP_DO_TYPE(T) \
  void read_simple(T& x); \
  void read_array(std::size_t, T *);
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

  virtual void read_string(std::size_t n, char* s);

protected:
  /// get the position in the XDR stream.
  uint32_t getPosition() const;

  /// set the position in the XDR stream.
  void setPosition(uint32_t pos);

  XDR xdr_; // the XDR stream
};


/** a dump for serializing objects into a file using the XDR format. */

class ALPS_DECL OXDRFileDump: public OXDRDump
{
public:
  /// open a new dump file with the given name
  // OXDRFileDump(const std::string& n);
  OXDRFileDump(const boost::filesystem::path& name, bool append=false);
  virtual ~OXDRFileDump();

  void flush();

private:
  // file reference and file name, needed by dump reference
  std::FILE* file_;

  /// open a file
  void open_file(const std::string&, bool=false);
};


/** a dump for deserializing objects from a file using the XDR format. */

class ALPS_DECL IXDRFileDump: public IXDRDump
{
public:
  /** open a file.
      @throws std::runtime_error if the file could not be openend. */
  // IXDRFileDump(const std::string& name);
  IXDRFileDump(const boost::filesystem::path& name);
  bool couldOpen() { return valid_;}

  virtual ~IXDRFileDump();

private:
  // file reference and name, needed by dump reference

  std::FILE* file_;
  bool valid_;

  void open_file(const std::string&);
};

} // end namespace alps

#endif // OSIRIS_XDRDUMP_H
