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

#ifndef ALPS_NGS_PARAMPROXY_HPP
#define ALPS_NGS_PARAMPROXY_HPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/detail/paramvalue.hpp>

#include <boost/function.hpp>
#include <boost/optional/optional.hpp>

#include <string>
#include <iostream>

namespace alps {

    class ALPS_DECL paramproxy {

        public:

            paramproxy(detail::paramvalue const & v)
                : value(v)
            {}

            paramproxy(
                  boost::function<detail::paramvalue()> const & g
                , boost::function<void(detail::paramvalue)> const & s
            )
                : getter(g)
                , setter(s)
            {}

            paramproxy(paramproxy const & arg)
                : value(arg.value)
                , getter(arg.getter)
                , setter(arg.setter)
            {}

			template<typename T> T cast() const {
				if (!defined)
					throw std::runtime_error(
						"No reference to parameter available" + ALPS_STACKTRACE
					);
				return (!!value ? getter() : *value).cast<T>();
			}

			#define ALPS_NGS_PARAMPROXY_MEMBER_DECL(T)								\
				operator T () const;												\
				paramproxy & operator=(T const & arg);								\
				T operator|( T v) const;
			ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMPROXY_MEMBER_DECL)
			#undef ALPS_NGS_PARAMPROXY_MEMBER_DECL

            void save(hdf5::archive & ar) const;
            void load(hdf5::archive &);

        private:

			bool defined;
            boost::optional<detail::paramvalue> value;
            boost::function<detail::paramvalue()> getter;
            boost::function<void(detail::paramvalue)> setter;
    };

    ALPS_DECL std::ostream & operator<<(std::ostream & os, paramproxy const &);

	#define ALPS_NGS_PARAMPROXY_ADD_OPERATOR_DECL(T)								\
		ALPS_DECL T operator+(param const & p, T s);								\
		ALPS_DECL T operator+(T s, param const & p);
	ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMPROXY_ADD_OPERATOR_DECL)
	#undef ALPS_NGS_PARAMPROXY_ADD_OPERATOR_DECL

    ALPS_DECL std::string operator+(param const & p, char const * s);
    ALPS_DECL std::string operator+(char const * s, param const & p);

}
	
#endif
