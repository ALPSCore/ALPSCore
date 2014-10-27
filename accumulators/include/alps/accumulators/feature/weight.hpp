/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_WEIGHT_HPP
#define ALPS_ACCUMULATOR_WEIGHT_HPP

#include <alps/accumulators/feature.hpp>
#include <alps/accumulators/parameter.hpp>
#include <alps/accumulators/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <boost/utility.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulators {

        template<typename T> class base_wrapper;

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

        template<typename T> base_wrapper<typename value_type<T>::type> const * weight(T const & arg) {
            return arg.weight();
        }

        namespace detail {

            template<typename A> typename boost::enable_if<
                  typename has_feature<A, weight_tag>::type
                , base_wrapper<typename value_type<A>::type> const *
            >::type weight_impl(A const & acc) {
                return weight(acc);
            }

            template<typename A> typename boost::disable_if<
                  typename has_feature<A, weight_tag>::type
                , base_wrapper<typename value_type<A>::type> const *
            >::type weight_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no weight-method" + ALPS_STACKTRACE);
                return NULL;
            }
        }

        namespace impl {

            template<typename T, typename B> class BaseWrapper<T, weight_tag, B> : public B {
                public:
                    virtual bool has_weight() const = 0;
                    virtual base_wrapper<T> const * weight() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, weight_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_weight() const { return has_feature<T, weight_tag>::type::value; }
                    base_wrapper<typename value_type<T>::type> const * weight() const { return detail::weight_impl(this->m_data); } 
            };

        }
    }
}

 #endif
