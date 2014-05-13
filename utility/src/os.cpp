/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#include <alps/utility/os.hpp>
//#include <alps/version.h> //FIXME
#include <boost/throw_exception.hpp>
#include <boost/filesystem/operations.hpp>
#include <stdexcept>
#include <vector>

#ifndef BOOST_POSIX_API
#include <windows.h>
#endif


#if defined(ALPS_HAVE_SYS_SYSTEMINFO_H)
# include <sys/systeminfo.h> // for sysinfo()
#elif defined(ALPS_HAVE_UNISTD_H)
# include <unistd.h>      // for gethostname() and getlogin()
# ifndef MAXHOSTNAMELEN
#   define MAXHOSTNAMELEN 256
# endif
#endif

#ifdef BOOST_NO_STDC_NAMESPACE
  namespace std { using ::system; }
#endif

namespace alps {

//=======================================================================
// hostname
//
// returns the host name
//-----------------------------------------------------------------------

std::string hostname()
{
#if defined(ALPS_HAVE_SYS_SYSTEMINFO_H)
  // the UNIX System V version
  char n[256];
  if(sysinfo(SI_HOSTNAME,n,256) < 0)
    boost::throw_exception(std::runtime_error("call to sysinfo failed in get_host_name"));
  return n;
#elif defined(ALPS_HAVE_UNISTD_H)
  // the BSD UNIX version
  char n[MAXHOSTNAMELEN];
  if(gethostname(n,MAXHOSTNAMELEN))
    boost::throw_exception(std::runtime_error("call to gethostname failed in get_host_name"));
  return n;
#else
  // the default version
  return "unnamed";
#endif
}

//=======================================================================
// username
//
// returns the username
//-----------------------------------------------------------------------

std::string username() {
#if defined(ALPS_HAVE_UNISTD_H)
  const char* login = getlogin();
  return (login ? std::string(getlogin()) : std::string("unknown"));
#else
  return std::string("unknown");
#endif
}


   // contributed by Jeff Flinn, from Boost 1.46
boost::filesystem::path temp_directory_path()
{
  using namespace boost::filesystem;
#   ifdef BOOST_POSIX_API
    const char* val = 0;
    
    (val = std::getenv("TMPDIR" )) ||
    (val = std::getenv("TMP"    )) ||
    (val = std::getenv("TEMP"   )) ||
    (val = std::getenv("TEMPDIR"));
    
    path p((val!=0) ? val : "/tmp");
    
    if (p.empty() || !is_directory(p))
      p=path(".");
      
    return p;
    
#   else  // Windows

    std::vector<TCHAR> buf(MAX_PATH);

    if (buf.empty() || GetTempPath(buf.size(), &buf[0])==0)
      return path(".");
        
    buf.pop_back();
    
    path p(buf.begin(), buf.end());
        
    if (!is_directory(p))
      p=path(".");

    
    return p;
#   endif
}

#ifdef ALPS_PREFIX
boost::filesystem::path installation_directory()
{
  return boost::filesystem::path(ALPS_PREFIX);
}
#else
boost::filesystem::path installation_directory()
{
  return boost::filesystem::path("");
}

#endif

boost::filesystem::path bin_directory()
{
  return installation_directory() / "bin";
}

  


} // end namespace alps
