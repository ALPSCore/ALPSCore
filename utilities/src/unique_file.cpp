/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/testing/unique_file.hpp>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <stdexcept>

// excerpt from <unistd.h>
extern "C" int close(int); 

namespace alps {
    namespace testing {

        /** @warning This method
         * cannot ensure that the file name is indeed unique by the
         * time the name is actually used.
         *
         * @warning Current implementation actually creates a file and then closes it.
         * 
         */
        unique_file::unique_file(const std::string& prefix, unique_file::action_type action) : action_(action)
        {
            // We need a modifiable 0-terminated char buffer
            std::vector<char> strbuf(prefix.begin(), prefix.end());
            strbuf.insert(strbuf.end(), 7, 'X');
            strbuf.back()='\0';

            int fd=mkstemp(&strbuf[0]);
            if (fd==-1) {
                throw std::runtime_error("Failed to generate a temporary name from template '"
                                             + std::string(&strbuf[0]) + "'");
            }
            close(fd);
            name_.assign(&strbuf[0]);
            if (REMOVE_NOW==action || REMOVE_AND_DISOWN==action) std::remove(&strbuf[0]);
        }

        unique_file::~unique_file() {
            if (REMOVE_AFTER==action_ || REMOVE_NOW==action_) {
                remove(name_.c_str()); // we ignore removal failure
            }
        }

    }
}
