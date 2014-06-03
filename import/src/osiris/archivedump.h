/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_OSIRIS_ARCHIVEDUMP_H
#define ALPS_OSIRIS_ARCHIVEDUMP_H

#include <alps/osiris/dump.h>
#include <iostream>

namespace alps {

/** A class to use a Boost output archive as an Osiris dump  */

template <class ARCHIVE> 
class archive_odump : public ODump
{
public:
  archive_odump (ARCHIVE& a) : archive_(a) {}    
  ~archive_odump() {}

/// INTERNAL ONLY
# define ALPS_DUMP_DO_TYPE(T) void write_simple(T x) { archive_ << x;}
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
    
  // write a C-style string
  void write_string(std::size_t, const char * x) { write_string(std::string(x));}
  void write_string(const std::string& s) { archive_.operator<<(s);}

private:
  ARCHIVE& archive_; // the Boost archive
};


/** A class to use a Boost input archive as an Osiris dump  */

template <class ARCHIVE> 
class archive_idump : public IDump
{
public:
  archive_idump (ARCHIVE& a) : archive_(a) {}    
  ~archive_idump() {}

/// INTERNAL ONLY
# define ALPS_DUMP_DO_TYPE(T) void read_simple(T& x) { archive_ >> x;}
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
      
  void read_string(std::size_t, char* s) {std::string y; archive_ >> y; std::strcpy(s,y.c_str());}
  void read_string(std::string& s) { archive_ >> s;}
  
private:
  ARCHIVE& archive_; // the Boost archive
};

} // end namespace alps

#endif // ALPS_OSIRIS_ARCHIVEDUMP_H
