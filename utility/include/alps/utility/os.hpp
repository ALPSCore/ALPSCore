/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_UTILITY_OS_HPP
#define ALPS_UTILITY_OS_HPP

//=======================================================================
// This file includes low level functions which depend on the OS used
//=======================================================================

#include <alps/config.hpp>
#include <boost/filesystem/path.hpp>
#include <string>

namespace alps {

/// returns the hostname
ALPS_DECL std::string hostname();

/// returns the username
ALPS_DECL std::string username();

/// returns the username
ALPS_DECL boost::filesystem::path temp_directory_path();

/// returns the installation directory
ALPS_DECL boost::filesystem::path installation_directory();

/// returns the program directory
ALPS_DECL boost::filesystem::path bin_directory();

} // end namespace

#endif // ALPS_UTILITY_OS_HPP
