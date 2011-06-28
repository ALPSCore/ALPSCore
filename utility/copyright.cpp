/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2011 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/config.h>
#include <alps/utility/copyright.hpp>
#include <alps/version.h>

void alps::print_copyright(std::ostream& out) {
  out << "based on the ALPS libraries version " << ALPS_VERSION << "\n";
  out << "  available from http://alps.comp-phys.org/\n";
  out << "  copyright (c) 1994-" << ALPS_YEAR
      << " by the ALPS collaboration.\n";
  out << "  Consult the web page for license details.\n";
  out << "  For details see the publication: \n"
      << "  B. Bauer et al., J. Stat. Mech. (2011) P05001.\n\n";
}

void alps::print_license(std::ostream& out) {
  out << "Please look at the file LICENSE.txt for the license conditions\n";
}

std::string alps::version() { return ALPS_VERSION; }

std::string alps::version_string() { return ALPS_VERSION_STRING; }

std::string alps::year() { return ALPS_YEAR; }

std::string alps::config_host() { return ALPS_CONFIG_HOST; }

std::string alps::config_user() { return ALPS_CONFIG_USER; }

std::string alps::compile_date() {
#if defined(__DATE__) && defined(__TIME__)
  return __DATE__ " " __TIME__;
#else
  return "unknown";
#endif
}
