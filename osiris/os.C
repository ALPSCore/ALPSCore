/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/osiris/os.h>

#include <boost/filesystem/operations.hpp>
#include <boost/limits.hpp>
#include <boost/throw_exception.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

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
  return std::string(getlogin());
#else
  return std::string("unknown");
#endif
}

} // end namespace alps
