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

#ifndef ALPS_NGS_PARAM_HPP
#define ALPS_NGS_PARAM_HPP

#warning this file is deprecated

#include <alps/ngs/config.hpp>
#include <alps/ngs/cast.hpp>
#include <alps/ngs/stacktrace.hpp>

#include <boost/function.hpp>
#include <boost/optional/optional.hpp>

#include <string>
#include <iostream>
#include <stdexcept>


/*
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

    // TODO: make possible to toke any base: std::map, python dict ...
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

    class ALPS_DECL param_proxy {

        public:

            param(param const & arg)
                : value_(arg.value_)
                , getter_(arg.getter_)
                , setter_(arg.setter_)
            {}

            param(std::string const & value)
                : value_(value)
            {}

            param(
                  boost::function<std::string()> const & getter
                , boost::function<void(std::string)> const & setter
            )
                : value_(boost::none_t())
                , getter_(getter)
                , setter_(setter)
            {}

			template<typename T> T cast() const {
				 return cast<T>(value_ == boost::none_t() ? getter_() : *value_);
            }

            template<typename T> operator T() const {
				return cast<T>();
            }

            template<typename T> param & operator=(T const & arg) {
                if (value_ != boost::none_t())
                    throw std::runtime_error("No reference available" + ALPS_STACKTRACE);
                setter_(cast<std::string>(arg));
                return *this;
            }
			
			save
			load

        private:

            boost::optional<std::string> value_;
            boost::function<std::string()> getter_;
            boost::function<void(std::string)> setter_;

    };

    ALPS_DECL std::ostream & operator<<(std::ostream & os, param const &);

	#define ALPS_NGS_PARAM_ADD_OPERATOR(T)											\
		ALPS_DECL T operator+(param const & p, T const & s);						\
		ALPS_DECL T operator+(T const & s, param const & p);
	ALPS_NGS_PARAM_ADD_OPERATOR(char)
    ALPS_NGS_PARAM_ADD_OPERATOR(signed char)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned char)
    ALPS_NGS_PARAM_ADD_OPERATOR(short)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned short)
    ALPS_NGS_PARAM_ADD_OPERATOR(int)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned)
    ALPS_NGS_PARAM_ADD_OPERATOR(long)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned long)
    ALPS_NGS_PARAM_ADD_OPERATOR(long long)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned long long)
    ALPS_NGS_PARAM_ADD_OPERATOR(float)
    ALPS_NGS_PARAM_ADD_OPERATOR(double)
    ALPS_NGS_PARAM_ADD_OPERATOR(long double)
    ALPS_NGS_PARAM_ADD_OPERATOR(bool)
    #undef ALPS_NGS_PARAM_ADD_OPERATOR

    ALPS_DECL std::string operator+(param const & p, char const * s);
    ALPS_DECL std::string operator+(char const * s, param const & p);

    ALPS_DECL std::string operator+(param const & p, std::string const & s);
    ALPS_DECL std::string operator+(std::string const & s, param const & p);

}
*/

namespace alps {

    class ALPS_DECL param {

        public:

            param(param const & arg)
                : value_(arg.value_)
                , getter_(arg.getter_)
                , setter_(arg.setter_)
            {}

            param(std::string const & value)
                : value_(value)
            {}

            param(
                  boost::function<std::string()> const & getter
                , boost::function<void(std::string)> const & setter
            )
                : value_(boost::none_t())
                , getter_(getter)
                , setter_(setter)
            {}

            template<typename T> operator T() const {
                return cast<T>(value_ == boost::none_t() ? getter_() : *value_);
            }

			template<typename T> T as() const {
				 return cast<T>(value_ == boost::none_t() ? getter_() : *value_);
            }

			operator std::string() const { return str(); }

            std::string str() const {
                return value_ == boost::none_t() ? getter_() : *value_;
            }

            template<typename T> param & operator=(T const & arg) {
                if (value_ != boost::none_t())
                    throw std::runtime_error("No reference available" + ALPS_STACKTRACE);
                setter_(cast<std::string>(arg));
                return *this;
            }

        private:

            boost::optional<std::string> value_;
            boost::function<std::string()> getter_;
            boost::function<void(std::string)> setter_;

    };

    ALPS_DECL std::ostream & operator<<(std::ostream & os, param const &);

	#define ALPS_NGS_PARAM_ADD_OPERATOR(T)											\
		ALPS_DECL T operator+(param const & p, T const & s);						\
		ALPS_DECL T operator+(T const & s, param const & p);
	ALPS_NGS_PARAM_ADD_OPERATOR(char)
    ALPS_NGS_PARAM_ADD_OPERATOR(signed char)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned char)
    ALPS_NGS_PARAM_ADD_OPERATOR(short)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned short)
    ALPS_NGS_PARAM_ADD_OPERATOR(int)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned)
    ALPS_NGS_PARAM_ADD_OPERATOR(long)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned long)
    ALPS_NGS_PARAM_ADD_OPERATOR(long long)
    ALPS_NGS_PARAM_ADD_OPERATOR(unsigned long long)
    ALPS_NGS_PARAM_ADD_OPERATOR(float)
    ALPS_NGS_PARAM_ADD_OPERATOR(double)
    ALPS_NGS_PARAM_ADD_OPERATOR(long double)
    ALPS_NGS_PARAM_ADD_OPERATOR(bool)
    #undef ALPS_NGS_PARAM_ADD_OPERATOR

    ALPS_DECL std::string operator+(param const & p, char const * s);
    ALPS_DECL std::string operator+(char const * s, param const & p);

    ALPS_DECL std::string operator+(param const & p, std::string const & s);
    ALPS_DECL std::string operator+(std::string const & s, param const & p);

}

#endif
