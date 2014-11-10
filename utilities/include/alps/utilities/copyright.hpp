/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

/// \file copyright.hpp
/// \brief prints copyright and license information
///
/// contains functions to print the license and copyright statements

#ifndef ALPS_COPYRIGHT_H
#define ALPS_COPYRIGHT_H

#include <iostream>
#include <alps/config.hpp>

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
