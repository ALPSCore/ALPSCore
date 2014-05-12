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

#ifndef ALPS_NGS_DETAIL_PARAMS_IMPL_BASE_HPP
#define ALPS_NGS_DETAIL_PARAMS_IMPL_BASE_HPP

#warning this file is deprecated

#include <string>

namespace alps {

    namespace detail {

        class params_impl_base {

            public:
                
                virtual ~params_impl_base() {};
            
                virtual std::size_t size() const = 0;

                virtual std::vector<std::string> keys() const = 0;

                virtual param operator[](std::string const &) = 0;

                virtual param const operator[](std::string const &) const = 0;

                virtual bool defined(std::string const &) const = 0;

                virtual void save(hdf5::archive &) const = 0;

                virtual void load(hdf5::archive &) = 0;
                
                virtual params_impl_base * clone() = 0;

                #ifdef ALPS_HAVE_MPI
                    virtual void broadcast(int root) = 0;
                #endif

        };

    }
}

#endif
