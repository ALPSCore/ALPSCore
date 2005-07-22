/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_OSIRIS_DUMPARCHIVE_H
#define ALPS_OSIRIS_DUMPARCHIVE_H

#include <alps/osiris/dump.h>
#include <alps/osiris/std/string.h>
#include <boost/archive/archive_exception.hpp>
#include <boost/archive/detail/iserializer.hpp>
#include <boost/archive/detail/interface_iarchive.hpp>
#include <boost/archive/detail/common_iarchive.hpp>
#include <boost/archive/detail/oserializer.hpp>
#include <boost/archive/detail/interface_oarchive.hpp>
#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/version.hpp>
#include <iostream>

namespace alps {

template <class T, class ARCHIVE>
T get(ARCHIVE& ar) { T x; ar >> x; return x;}

// A class to use an Osiris dump as Boost output archive

class odump_archive 
 : public boost::archive::detail::common_oarchive<odump_archive>
{
public:
  odump_archive(ODump& d, bool c=true) : 
#if (BOOST_VERSION >= 103300)
    boost::archive::detail::common_oarchive<odump_archive>(0),
#endif
    dump_(d), compatible_(c) {}
//  template<class T>
//  odump_archive& operator<<(const T & t) 
//  { boost::serialization::save(* This(), t); return * This(); }

  // archives are expected to support this function
  void save_binary(const void *address, size_t count)
  { dump_.write_array(count,reinterpret_cast<const char*>(address)); }

  friend class boost::archive::save_access;
  template<class T> void save(const T & t);

# define ALPS_DUMP_DO_TYPE(T) void save(T x) { dump_ << x;}
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
  ALPS_DUMP_DO_TYPE(const std::string&)
# undef ALPS_DUMP_DO_TYPE
    // any datatype not specifed below will be handled
    // by this function
    template<class T>
    void save_override(const T & t, BOOST_PFTO int)
    {
        boost::archive::save(* this->This(), t);
    }
    // binary files don't include the optional information 
    void save_override(const boost::archive::class_id_optional_type & /* t */, int){}

    void save_override(const boost::archive::version_type & t, int){
      if (!compatible_) {  // upto 255 versions
        // note:t.t resolves borland ambguity
        uint16_t x = t.t;
        * this->This() << x;
      }
    }
    void save_override(const boost::archive::class_id_type & t, int){
      if (!compatible_) {   // upto 32K classes
        int_least16_t x = t.t;
        * this->This() << x;
      }
    }
    void save_override(const boost::archive::class_id_reference_type & t, int){
      if (!compatible_) {  // upto 32K classes
        int_least16_t x = t.t;
        * this->This() << x;
      }
    }
    void save_override(const boost::archive::object_id_type & t, int){
      if (!compatible_) {  // upto 2G objects
        uint_least32_t x = t.t;
        * this->This() << x;
      }
    }
    void save_override(const boost::archive::object_reference_type & t, int){
        // upto 2G objects
        uint_least32_t x = t.t;
        * this->This() << x;
    }
    void save_override(const boost::archive::tracking_type & t, int){
      if (!compatible_) {
        char x = t.t;
        * this->This() << x;
      }
    }

    // explicitly convert to char * to avoid compile ambiguities
    void save_override(const boost::archive::class_name_type & t, int){
#if (BOOST_VERSION >= 103300)
        this->This()->save(std::string(static_cast<const char *>(t)));
#else
        * this->This() << std::string(static_cast<const char *>(t));
#endif
    }

private:    
  ODump& dump_;
  bool compatible_;
};

// A class to use an Osiris dump as Boost input archive 

class idump_archive 
 : public boost::archive::detail::common_iarchive<idump_archive>
{
public:
  idump_archive(IDump& d, bool c=true) : 
#if (BOOST_VERSION >= 103300)
    boost::archive::detail::common_iarchive<idump_archive>(0), 
#endif
    dump_(d), compatible_(c) {}
  
//  template<class T>
//  idump_archive& operator>>(T & t) 
//  { boost::serialization::load(* This(), t); return * This(); }

  // archives are expected to support this function
  void load_binary(void *address, size_t count)
  { dump_.read_array(count,reinterpret_cast<char*>(address)); }

  friend class boost::archive::load_access;
  template<class T> void load(T & t);

# define ALPS_DUMP_DO_TYPE(T) void load(T& x) { dump_ >> x;}
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
  void load (std::string& s) { dump_.read_string(s);}

    // intermediate level to support override of operators
    // fot templates in the absence of partial function 
    // template ordering
    template<class T>
    void load_override(T & t, BOOST_PFTO int)
    {
        boost::archive::load(* this->This(), t);
    }

    // binary files don't include the optional information 
    void load_override(boost::archive::class_id_optional_type &, int){}

    // the following have been overridden to provide specific sizes
    // for these pseudo prmitive types.
    void load_override(boost::archive::version_type & t, int){ 
      if (!compatible_) { // upto 255 versions
        uint16_t x;
        * this->This() >> x;
        t = boost::archive::version_type(x);
      }
      else
        t  = boost::archive::version_type(dump_.version());
    }
    void load_override(boost::archive::class_id_type & t, int){
      if (!compatible_) {   // upto 32K classes
        int_least16_t x;
        * this->This() >> x;
        t = boost::archive::class_id_type(x);
      }
    }
    void load_override(boost::archive::class_id_reference_type & t, int){
      if (!compatible_) {   // upto 32K classes
        int_least16_t x;
        * this->This() >> x;
        t = boost::archive::class_id_reference_type(x);
      }
    }
    void load_override(boost::archive::object_id_type & t, int){
      if (!compatible_) {   // upto 2G objects
        uint_least32_t x;
        * this->This() >> x;
        t = boost::archive::object_id_type(x);
      }
    }
    void load_override(boost::archive::object_reference_type & t, int){
      if (!compatible_) {   // upto 2G objects
        uint_least32_t x;
        * this->This() >> x;
        t = boost::archive::object_reference_type(x);
      }
    }
    void load_override(boost::archive::tracking_type & t, int){
      if (!compatible_) { 
        char x;
        * this->This() >> x;
        t = (0 != x);
      }
    }

    void load_override(boost::archive::class_name_type & t, int){
      if (!compatible_) { 
        std::string cn;
        cn.reserve(BOOST_SERIALIZATION_MAX_KEY_SIZE);
        load_override(cn, 0);
        if(cn.size() > (BOOST_SERIALIZATION_MAX_KEY_SIZE - 1))
            boost::throw_exception(
                boost::archive::archive_exception(boost::archive::archive_exception::invalid_class_name)
           );
        std::memcpy(t, cn.data(), cn.size());
        // .t is a borland tweak
        t.t[cn.size()] = '\0';
      }
    }

private:    

  IDump& dump_;
  bool compatible_;
};

} // end namespace alps
/*
#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif


template <class T>
inline alps::ODump& operator<<(alps::ODump& dump, const T& x)
{
  odump_archive ar(dump);
  ar << x;
  return dump;
}

template <class T>
inline alps::IDump& operator>>(alps::IDump& dump, T& x)
{
  idump_archive ar(dump);
  ar >> x;
  return dump;
}


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif
*/

#endif // ALPS_OSIRIS_DUMPARCHIVE_H
