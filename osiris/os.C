/***************************************************************************
* PALM++/palm library
*
* palm/os.C
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

#include <alps/osiris/os.h>

#include <boost/filesystem/operations.hpp>
#include <boost/limits.hpp>
#include <boost/throw_exception.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

#include <errno.h>

#ifdef ALPS_HAVE_SYS_TIME_H
# include <sys/time.h>
#endif

#if defined(ALPS_HAVE_SYS_SYSTEMINFO_H)
# include <sys/systeminfo.h> // for sysinfo()
#elif defined(ALPS_HAVE_UNISTD_H)
# include <unistd.h>      // for gethostname()
# ifndef MAXHOSTNAMELEN
#   define MAXHOSTNAMELEN 256
# endif
#endif

#ifdef BOOST_NO_STDC_NAMESPACE
  namespace std { using ::system; }
#endif

namespace alps {
namespace detail {

//=======================================================================
// initialization of global variables contained in global object
//-----------------------------------------------------------------------
class os_variables {
public:
  os_variables();
  double old_clock;
  double clock_offset;
};

extern os_variables the_os_vars;

//=======================================================================
// initialization of global variables contained in global object
//-----------------------------------------------------------------------

os_variables::os_variables()
{
  timeval tp;
  gettimeofday(&tp,0);
  old_clock=tp.tv_sec+1e-6*tp.tv_usec;
  clock_offset=0.;
}

os_variables the_os_vars;

} // end namespace detail


//=======================================================================
// get_host_name
// 
// returns the Host name
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
// dclock & dclock_prec
// 
// <dclock> returns the time in seconds, with a precision returned by
// <dclock_prec>. The time can be either CPU time or wallclock time,
// depending on the architecture.
//-----------------------------------------------------------------------

double dclock(void)
{
  timeval tp;
  gettimeofday(&tp,0);
  return tp.tv_sec - detail::the_os_vars.old_clock + 1e-6*tp.tv_usec;
}

double dclock_resolution(void)
{
  return 1./double(CLOCKS_PER_SEC);
}

} // end namespace alps
