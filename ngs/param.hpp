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

#include <alps/ngs/macros.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/convert.hpp>

#include <boost/function.hpp>
#include <boost/optional/optional.hpp>

#include <string>
#include <iostream>

namespace alps {

    class param {

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
                return convert<T>(value_ == boost::none_t() ? getter_() : *value_);
            }

            std::string str() const {
                return value_ == boost::none_t() ? getter_() : *value_;
            }

            template<typename T> param & operator=(T const & arg) {
                if (value_ != boost::none_t())
                    ALPS_NGS_THROW_RUNTIME_ERROR("No reference available");
                setter_(convert<std::string>(arg));
                return *this;
            }

        private:

            boost::optional<std::string> value_;
            boost::function<std::string()> getter_;
            boost::function<void(std::string)> setter_;

    };

    std::ostream & operator<<(std::ostream & os, param const &);

    
    
	#define ALPS_NGS_PARAM_ADD_OPERATOR(T)								\
		T operator+(param const & p, T const & s);						\
		T operator+(T const & s, param const & p);
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


    std::string operator+(param const & p, std::string const & s);

    std::string operator+(std::string const & s, param const & p);

}

#endif
