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

#ifndef ALPS_NGS_HDF5_ERROR_HPP
#define ALPS_NGS_HDF5_ERROR_HPP

#include <string>
#include <stdexcept>

namespace alps {
    namespace hdf5 {

        class archive_error : public std::runtime_error {
            public:
                archive_error(std::string const & what)
                    : std::runtime_error(what) 
                {}
        };

        #define DEFINE_ALPS_HDF5_EXCEPTION(name)                                    \
            class name : public archive_error {                                     \
                public:                                                             \
                    name (std::string const & what)                                 \
                        : archive_error(what)                                       \
                    {}                                                              \
            };
        DEFINE_ALPS_HDF5_EXCEPTION(archive_not_found)
        DEFINE_ALPS_HDF5_EXCEPTION(archive_closed)
        DEFINE_ALPS_HDF5_EXCEPTION(invalid_path)
        DEFINE_ALPS_HDF5_EXCEPTION(path_not_found)
        DEFINE_ALPS_HDF5_EXCEPTION(wrong_type)
        #undef DEFINE_ALPS_HDF5_EXCEPTION
    }
};

#endif
