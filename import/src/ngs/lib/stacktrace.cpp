/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <alps/ngs/stacktrace.hpp>

#ifndef ALPS_NGS_NO_STACKTRACE

#include <sstream>

#include <cxxabi.h>
#include <stdlib.h>
#include <execinfo.h>

#endif

namespace alps {
    namespace ngs {

#ifndef ALPS_NGS_NO_STACKTRACE

        // TODO: ues boost::units::detail::demangle
        // in #include <boost/units/detail/utility.hpp>
        std::string stacktrace() {
            std::ostringstream buffer;
            void * stack[ALPS_NGS_MAX_FRAMES + 1];
            std::size_t depth = backtrace(stack, ALPS_NGS_MAX_FRAMES + 1);
            if (!depth)
                buffer << "  <empty, possibly corrupt>" << std::endl;
            else {
                char * * symbols = backtrace_symbols(stack, depth);
                for (std::size_t i = 1; i < depth; ++i) {
                    std::string symbol = symbols[i];
                    // TODO: use alps::ngs::stacktrace to find the position of the demangling name
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
}
