/***************************************************************************
* PALM++/palm library
*
* palm/os.h
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

#ifndef PALM_OS_H
#define PALM_OS_H

//=======================================================================
// This file includes low level functions which depend on the OS used
//=======================================================================

#include <alps/config.h>
#include <string>

namespace alps {

/// set the string to contain the name of the host
std::string hostname();

/** the clock in seconds.
    A fast system call to obtain a clock. Depending on the
    system this can be either wallclock time or CPU time. 
    The rationale for this function is that on some MPP systems the
    clock() function is synchronized over all nodes, making it
    a very resource intensive call. The function also
    handles overflows of the clock, as long as it is
    called at least once during the period of the clock. */
double dclock();

/// the resolution of the dclock() function
double dclock_resolution();

} // end namespace

#endif // PALM_OS_H
