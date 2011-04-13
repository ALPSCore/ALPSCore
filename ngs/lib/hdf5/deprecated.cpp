
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

#include <alps/ngs/hdf5/deprecated.hpp>

namespace alps {
    namespace hdf5 {

        #define ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS(T)                                                                                                                 \
            void save(                                                                                                                                                 \
                  oarchive & ar                                                                                                                                        \
                , std::string const & path                                                                                                                             \
                , T const & value                                                                                                                                      \
                , std::vector<std::size_t> size                                                                                                                        \
                , std::vector<std::size_t> chunk                                                                                                                       \
                , std::vector<std::size_t> offset                                                                                                                      \
            ) {                                                                                                                                                        \
                save(static_cast<archive &>(ar), path, value, size, chunk, offset);                                                                                    \
            }                                                                                                                                                          \
                                                                                                                                                                       \
            void load(                                                                                                                                                 \
                  iarchive & ar                                                                                                                                        \
                , std::string const & path                                                                                                                             \
                , T & value                                                                                                                                            \
                , std::vector<std::size_t> chunk                                                                                                                       \
                , std::vector<std::size_t> offset                                                                                                                      \
            ) {                                                                                                                                                        \
                load(static_cast<archive &>(ar), path, value, chunk, offset);                                                                                          \
            }
        ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS)

        #undef ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS
    }
}
