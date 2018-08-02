/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
// Use ALPSCore for something
#include <alps/config.hpp>
#include <alps/utilities/stringify.hpp>

// Let's use, e.g.,  boost::filesystem
#include <boost/filesystem.hpp>

int main(int argc, char **argv)
{
    std::cout << "Using ALPSCore version " << ALPS_STRINGIFY(ALPSCORE_VERSION);
    if (! argc>0) {
        std::cerr << "Should not happen: program name is not available.\n";
        return 1;
    }

    namespace fs=boost::filesystem;

    // Get the name of this program file:
    fs::path myname(argv[0]);

    // Determine the size of the program file:
    auto fsize=fs::file_size(myname);

    std::cout << "\nThis program is called " << myname.string()
              << " and its size is " << fsize
              << "\n";
    return 0;
}
