/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/ngs/detail/paramproxy.hpp>

#include <alps/numeric/vector_functions.hpp>

#include <stdexcept>

namespace alps {
    namespace detail {

        void paramproxy::save(hdf5::archive & ar) const {
            if (!defined)
                throw std::runtime_error("No reference to parameter '" + key + "' available" + ALPS_STACKTRACE);
            ar[""] << (!!value ? *value : getter());
        }

        void paramproxy::load(hdf5::archive & ar) {
            if (!defined || !!value)
                throw std::runtime_error("No reference to parameter '" + key + "' available" + ALPS_STACKTRACE);
            if (!!value) {
                detail::paramvalue value;
                ar[""] >> value;
                setter(value);
            } else
                ar[""] >> *value;
        }

        void paramproxy::print(std::ostream & os) const {
			if (!defined)
				throw std::runtime_error("No parameter '" + key + "' available" + ALPS_STACKTRACE);
			os << (!value ? getter() : *value);
        }

        std::ostream & operator<<(std::ostream & os, paramproxy const & v) {
			v.print(os);
            return os;
		}

        #define ALPS_NGS_PARAMPROXY_ADD_OPERATOR_IMPL(T)                                \
            T operator+(paramproxy const & p, T s) {                                    \
                using boost::numeric::operators::operator+=;                            \
                return s += p.cast< T >();                                              \
            }                                                                           \
            T operator+(T s, paramproxy const & p) {                                    \
                using boost::numeric::operators::operator+=;                            \
                return s += p.cast< T >();                                              \
            }
        ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMPROXY_ADD_OPERATOR_IMPL)
        #undef ALPS_NGS_PARAMPROXY_ADD_OPERATOR_IMPL

        std::string operator+(paramproxy const & p, char const * s) {
            return p.cast<std::string>() + s;
        }

        std::string operator+(char const * s, paramproxy const & p) {
            return s + p.cast<std::string>();
        }

    }
}
