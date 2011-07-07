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

#ifndef ALPS_NGS_PARAMS_HPP
#define ALPS_NGS_PARAMS_HPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/param.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/detail/params_impl_base.hpp>

#include <boost/scoped_ptr.hpp>

#ifdef ALPS_HAVE_PYTHON
    #include <alps/ngs/boost_python.hpp>
#endif

#include <string>

namespace alps {

    namespace detail {

        class params_impl_base;

    }

    class params {

        public:

            params(params const &);

            params(hdf5::archive &);

            params(std::string const &);

            #ifdef ALPS_HAVE_PYTHON
                params(boost::python::object const & arg);
            #endif

            virtual ~params();

            std::size_t size() const;

            std::vector<std::string> keys() const;

            param operator[](std::string const &);

            param const operator[](std::string const &) const;

            template<typename T> param value_or_default(std::string const & key, T const & value) const {
                return defined(key) 
                    ? operator[](key) 
                    : param(convert<std::string>(value))
                ;
            }

            bool defined(std::string const &) const;

            void save(hdf5::archive &) const;

            void load(hdf5::archive &);
            
            #ifdef ALPS_HAVE_PYTHON
                // USE FOR PYTHON EXPORT ONLY!
                detail::params_impl_base * get_impl();
                detail::params_impl_base const * get_impl() const;
            #endif
            
            // TODO: add boost serialization support

        private:

            boost::scoped_ptr<detail::params_impl_base> impl_;

    };

}

#endif
