/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: convert2xml.C 3523 2009-12-12 05:52:24Z troyer $ */
#include <alps/config.h>

#include <string>

namespace alps {

  /// convert a file from XDR format to XML
  std::string ALPS_DECL convert2xml(std::string const& name);
} // end namespace
