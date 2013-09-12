/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_UTILITY_OS_HPP
#define ALPS_UTILITY_OS_HPP

//=======================================================================
// This file includes low level functions which depend on the OS used
//=======================================================================

#include <alps/config.h>
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
