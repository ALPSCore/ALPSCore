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

#ifndef ALPS_NGS_MCPARAMS_HPP
#define ALPS_NGS_MCPARAMS_HPP

#include <alps/ngs/hdf5.hpp>

#include <alps/config.h>

#include <boost/variant.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>
// use alps::convert
#include <boost/lexical_cast.hpp>

#include <map>
#include <vector>
#include <string>

namespace alps {

    namespace detail {
        typedef boost::mpl::vector<std::string, int, double> mcparamvalue_types;
        typedef boost::make_variant_over<mcparamvalue_types>::type mcparamvalue_base;
    }

    class mcparamvalue : public detail::mcparamvalue_base {

        public:

            mcparamvalue() {}

            template <typename T> mcparamvalue(T const & v): detail::mcparamvalue_base(v) {}

            mcparamvalue(mcparamvalue const & v): detail::mcparamvalue_base(static_cast<detail::mcparamvalue_base const &>(v)) {}

            std::string str() const;

            template <typename T> typename boost::enable_if<
                  typename boost::mpl::contains<detail::mcparamvalue_types, T>::type
                , mcparamvalue &
            >::type operator=(T const & v) {
                detail::mcparamvalue_base::operator=(v);
                return *this;
            }

            template <typename T> typename boost::disable_if<
                  typename boost::mpl::contains<detail::mcparamvalue_types, T>::type
                , mcparamvalue &
            >::type operator=(T const & v) {
                detail::mcparamvalue_base::operator=(boost::lexical_cast<std::string>(v));
                return *this;
            }

            #define ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(T)    \
                operator T () const;
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(short)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(unsigned short)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(int)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(unsigned int)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(long)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(unsigned long)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(long long)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(unsigned long long)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(float)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(double)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(long double)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(bool)
            ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL(std::string)
            #undef ALPS_NGS_MCPARAMS_CAST_OPERATOR_DECL
    };

    // TODO: can we keep parameter ordering? like in curent class
    class mcparams : public std::map<std::string, mcparamvalue> {

        public: 

            mcparams(std::string const & input_file);

            mcparamvalue & operator[](std::string const & k);

            mcparamvalue const & operator[](std::string const & k) const;

            mcparamvalue value_or_default(std::string const & k, mcparamvalue const & v) const;

            bool defined(std::string const & k) const;

            void save(hdf5::archive & ar) const;

            void load(hdf5::archive & ar);

    };
}

#endif
