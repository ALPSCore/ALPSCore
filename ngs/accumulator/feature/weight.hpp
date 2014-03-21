/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                              Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_ACCUMULATOR_WEIGHT_HPP
#define ALPS_NGS_ACCUMULATOR_WEIGHT_HPP

#include <alps/ngs/accumulator/feature.hpp>
#include <alps/ngs/accumulator/parameter.hpp>
#include <alps/ngs/accumulator/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <boost/utility.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulator {

        class base_wrapper;

        // this should be called namespace tag { struct weight; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct weight_tag;

        template<typename T> struct has_feature<T, weight_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename C> static char check(boost::integral_constant<std::size_t, sizeof(helper(&C::owns_weight))>*);
            template<typename C> static double check(...);
            typedef boost::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
        };

        namespace detail {
            struct no_weight_type {};
            template <bool, typename T> struct weight_type_impl {
                typedef no_weight_type type;
            };
            template <typename T> struct weight_type_impl<true, T> {
                typedef typename T::weight_type type;
            };
        }

        template<typename T> struct weight_type {
            typedef typename detail::weight_type_impl<has_feature<T, weight_tag>::type::value, T>::type type;
        };

        template<typename T> base_wrapper const * weight(T const & arg) {
            return arg.weight();
        }

        namespace detail {

            template<typename A> typename boost::enable_if<
                  typename has_feature<A, weight_tag>::type
                , base_wrapper const *
            >::type weight_impl(A const & acc) {
                return weight(acc);
            }

            template<typename A> typename boost::disable_if<
                  typename has_feature<A, weight_tag>::type
                , base_wrapper const *
            >::type weight_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no weight-method" + ALPS_STACKTRACE);
                return NULL;
            }
        }

        namespace impl {

            template<typename B> class BaseWrapper<weight_tag, B> : public B {
                public:
                    virtual bool has_weight() const = 0;
            };

            template<typename T, typename B> class ResultTypeWrapper<T, weight_tag, B> : public B {
                public:
                    virtual base_wrapper const * weight() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, weight_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_weight() const { return has_feature<T, weight_tag>::type::value; }

                    base_wrapper const * weight() const { return detail::weight_impl(this->m_data); }
            };

        }
    }
}

 #endif
