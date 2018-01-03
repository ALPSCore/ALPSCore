/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/stacktrace.hpp>

#ifndef ALPS_UTILITY_NO_STACKTRACE

#include <sstream>

#include <cxxabi.h>
#include <stdlib.h>
#include <execinfo.h>

#endif

namespace alps {

#ifndef ALPS_UTILITY_NO_STACKTRACE

    // TODO: use boost::units::detail::demangle
    // in #include <boost/units/detail/utility.hpp>
    std::string stacktrace() {
        std::ostringstream buffer;
        void * stack[ALPS_MAX_FRAMES + 1];
        std::size_t depth = backtrace(stack, ALPS_MAX_FRAMES + 1);
        if (!depth)
            buffer << "  <empty, possibly corrupt>" << std::endl;
        else {
            char * * symbols = backtrace_symbols(stack, depth);
            for (std::size_t i = 1; i < depth; ++i) {
                std::string symbol = symbols[i];
                // TODO: use alps::stacktrace to find the position of the demangling name
                if (symbol.find_first_of(' ', 59) != std::string::npos) {
                    std::string name = symbol.substr(59, symbol.find_first_of(' ', 59) - 59);
                    int status;
                    char * demangled = abi::__cxa_demangle(name.c_str(), NULL, NULL, &status);
                    if (!status) {
                        buffer << "    " 
                               << symbol.substr(0, 59) 
                               << demangled
                               << symbol.substr(59 + name.size())
                               << std::endl;
                        free(demangled);
                    } else
                        buffer << "    " << symbol << std::endl;
                } else
                    buffer << "    " << symbol << std::endl;
            }
            free(symbols);
        }
        return buffer.str();
    }

#else

    std::string stacktrace() {
        return "";
    }

#endif

}
