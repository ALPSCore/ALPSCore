/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2010 by Matthias Troyer <troyer@comp-phys.org>,
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

/// \file copyright.h
/// \brief prints copyright and license information
///
/// contains functions to print the license and copyright statements

#ifndef ALPS_COPYRIGHT_H
#define ALPS_COPYRIGHT_H

#include <iostream>
#include <alps/config.h>

namespace alps {

/// print the ALPS library copyright statement
/// \param out the output stream to which the copyright statement should be written
ALPS_DECL void print_copyright(std::ostream& out);

/// print the ALPS license
/// \param out the output stream to which the license should be written
ALPS_DECL void print_license(std::ostream& out);

/// return ALPS version
ALPS_DECL std::string version();

/// return ALPS version (full string)
ALPS_DECL std::string version_string();

/// return latest publish year of ALPS
ALPS_DECL std::string year();

/// return the hostname where configure script was executed
ALPS_DECL std::string config_host();

/// return the username who executed configure script
ALPS_DECL std::string config_user();

/// return the compile date of ALPS
ALPS_DECL std::string compile_date();

} // end namespace alps

#endif
