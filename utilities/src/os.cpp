/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#include"boost/asio/ip/host_name.hpp"
#include"boost/filesystem.hpp"
namespace alps {

//=======================================================================
// hostname
//
// returns the host name
//-----------------------------------------------------------------------

std::string hostname()
{
  return boost::asio::ip::host_name();
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


boost::filesystem::path temp_directory_path()
{
  return boost::filesystem::temp_directory_path();
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
