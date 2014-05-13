/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
