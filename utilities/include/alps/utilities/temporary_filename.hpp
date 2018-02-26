/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITIES_TEMPORARY_FILENAME_HPP_ae2c8128ef7647a8827212faaab05d00
#define ALPS_UTILITIES_TEMPORARY_FILENAME_HPP_ae2c8128ef7647a8827212faaab05d00

#include <alps/testing/unique_file.hpp>

namespace alps {
    /// Generate a reasonably unique file name with a given prefix
    /**
       @note There is no guarantee that the name rename unique by the
       time it is used.

       @note The file is actually created (then removed), avoid
       generating many temporary file names in a loop.
    */
    inline std::string temporary_filename(const std::string& prefix)
    {
        return alps::testing::unique_file(prefix, alps::testing::unique_file::REMOVE_AND_DISOWN).name();
    }
}
#endif /* ALPS_UTILITIES_TEMPORARY_FILENAME_HPP_ae2c8128ef7647a8827212faaab05d00 */

