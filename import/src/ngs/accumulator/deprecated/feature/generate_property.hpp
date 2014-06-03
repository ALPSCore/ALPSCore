/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_DETAIL_GENERATE_PROPERTY
#define ALPS_NGS_ALEA_DETAIL_GENERATE_PROPERTY

#include <alps/ngs/stacktrace.hpp>

#include <alps/ngs/alea/feature/value_type.hpp>

#include <boost/utility.hpp>
#include <boost/type_traits.hpp>

#include <sstream>
#include <typeinfo>
#include <stdexcept>

namespace alps {
    namespace accumulator {

        template<typename T, typename Tag> struct has_feature {};
        template <typename base_type, typename Tag> class feature_accumulator_property {};
        template <typename base_type, typename Tag> class feature_result_property {};

        namespace detail {
            template<typename Accum, typename Tag, bool> struct feature_accumulator_property_impl {};
            template<typename Result, typename Tag, bool> struct feature_result_property_impl {};
        }
    }
}

#define GENERATE_PROPERTY(FNNAME, TAG)                                                                                      \
                                                                                                                            \
    /* = = = = = = = = = = I N F O   T R A I T = = = = = = = = = = */                                                       \
    template<typename T> struct has_feature< TAG , T> {                                                                     \
        template <typename U, void (U::*)() > struct helper {};                                                             \
        template<typename U> static char check(helper<U, &U:: FNNAME >*);                                                   \
        template<typename U> static double check(...);                                                                      \
        enum { value = (sizeof(char) == sizeof(check<T>(0))) };                                                             \
    };                                                                                                                      \
                                                                                                                            \
    /* = = = = = = = = = = F C T   V I A   M E M B E R = = = = = = = = = = */                                               \
                                                                                                                            \
    template <typename Accum> inline typename FNNAME ## _type<typename alps::accumulator::value_type<Accum>::type>::type FNNAME (              \
        Accum const & arg                                                                                                   \
    ) {                                                                                                                     \
        return arg. FNNAME ();                                                                                              \
    }                                                                                                                       \
                                                                                                                            \
    namespace detail {                                                                                                      \
                                                                                                                            \
    /* = = = = = = = = = = A V O I D S   N A M E C O N F L I C T S = = = = = = = = = = */                                   \
                                                                                                                            \
        template<typename Accum>                                                                                            \
        inline typename FNNAME ## _type<typename alps::accumulator::value_type<Accum>::type>::type FNNAME ## _impl(Accum const & arg) {        \
            return FNNAME (arg);                                                                                            \
        }                                                                                                                   \
                                                                                                                            \
    /* = = = = = = = = = = P R O P E R T Y   I M P L   W I T H   F C T = = = = = = = = = = */                               \
                                                                                                                            \
        template <typename base_type> class feature_accumulator_property_impl< TAG , base_type, true>: public base_type {   \
            public:                                                                                                         \
                feature_accumulator_property_impl(): base_type() {}                                                         \
                feature_accumulator_property_impl(typename base_type::accum_type const & arg): base_type(arg) {}            \
                bool has_ ## FNNAME () const { return true; }                                                               \
                typename FNNAME ## _type<                                                                                   \
				typename alps::accumulator::value_type<typename base_type::accum_type>::type                                               \
                >::type FNNAME() const {                                                                                    \
                    return FNNAME ## _impl(base_type::accum_);                                                              \
                }                                                                                                           \
        };                                                                                                                  \
                                                                                                                            \
        template <typename base_type> class feature_result_property_impl< TAG , base_type, true>: public base_type {        \
            public:                                                                                                         \
                feature_result_property_impl(): base_type() {}                                                              \
                feature_result_property_impl(typename base_type::result_type const & arg): base_type(arg) {}                \
                bool has_ ## FNNAME () const { return true; }                                                               \
                typename FNNAME ## _type<                                                                                   \
				typename alps::accumulator::value_type<typename base_type::result_type>::type                                              \
                >::type FNNAME() const {                                                                                    \
                    return FNNAME ## _impl(base_type::result_);                                                             \
                }                                                                                                           \
        };                                                                                                                  \
                                                                                                                            \
    /* = = = = = = = = = = P R O P E R T Y   I M P L   W I T H O U T   F C T = = = = = = = = = = */                         \
                                                                                                                            \
        template <typename base_type> class feature_accumulator_property_impl< TAG , base_type, false>: public base_type {  \
            public:                                                                                                         \
                feature_accumulator_property_impl(): base_type() {}                                                         \
                feature_accumulator_property_impl(typename base_type::accum_type const & arg): base_type(arg) {}            \
                bool has_ ## FNNAME() const { return false; }                                                               \
                typename FNNAME ## _type<                                                                                   \
                    typename alps::accumulator::value_type<typename base_type::accum_type>::type                                               \
                >::type FNNAME () const {                                                                                   \
                    throw std::runtime_error(                                                                               \
                        std::string(typeid(typename base_type::accum_type).name()) + " has no " + #FNNAME + "-method"       \
                        + ALPS_STACKTRACE                                                                                   \
                    );                                                                                                      \
                    return typename FNNAME ## _type<typename alps::accumulator::value_type<typename base_type::accum_type>::type>::type();     \
                }                                                                                                           \
        };                                                                                                                  \
                                                                                                                            \
        template <typename base_type> class feature_result_property_impl< TAG , base_type, false>: public base_type {       \
            public:                                                                                                         \
                feature_result_property_impl(): base_type() {}                                                              \
                feature_result_property_impl(typename base_type::result_type const & arg): base_type(arg) {}                \
                bool has_ ## FNNAME() const { return false; }                                                               \
                typename FNNAME ## _type<                                                                                   \
                    typename alps::accumulator::value_type<typename base_type::result_type>::type                                              \
                >::type FNNAME () const {                                                                                   \
                    throw std::runtime_error(                                                                               \
                        std::string(typeid(typename base_type::result_type).name()) + " has no " + #FNNAME + "-method"      \
                        + ALPS_STACKTRACE                                                                                   \
                    );                                                                                                      \
                    return typename FNNAME ## _type<typename alps::accumulator::value_type<typename base_type::result_type>::type>::type();    \
                }                                                                                                           \
        };                                                                                                                  \
    }                                                                                                                       \
                                                                                                                            \
    /* = = = = = = D E R I V E   F R O M   T H E   R I G H T   F C T   I M P  L  = = = = = = */                             \
                                                                                                                            \
    template <typename base_type> class feature_accumulator_property< TAG, base_type>                                       \
        : public detail::feature_accumulator_property_impl<TAG, base_type, has_feature<                                     \
              TAG, typename base_type::accum_type                                                                           \
          >::value>                                                                                                         \
    {                                                                                                                       \
        public:                                                                                                             \
            feature_accumulator_property()                                                                                  \
                : detail::feature_accumulator_property_impl< TAG, base_type, has_feature<                                   \
                      TAG, typename base_type::accum_type                                                                   \
                  >::value>()                                                                                               \
            {}                                                                                                              \
                                                                                                                            \
            feature_accumulator_property(typename base_type::accum_type const & acc)                                        \
                : detail::feature_accumulator_property_impl< TAG, base_type, has_feature<                                   \
                      TAG, typename base_type::accum_type                                                                   \
                  >::value>(acc)                                                                                            \
            {}                                                                                                              \
    };                                                                                                                      \
                                                                                                                            \
    template <typename base_type> class feature_result_property< TAG, base_type>                                            \
        : public detail::feature_result_property_impl<TAG, base_type, has_feature<                                          \
              TAG, typename base_type::result_type                                                                          \
          >::value>                                                                                                         \
    {                                                                                                                       \
        public:                                                                                                             \
            feature_result_property()                                                                                       \
                : detail::feature_result_property_impl< TAG, base_type, has_feature<                                        \
                      TAG, typename base_type::result_type                                                                  \
                  >::value>()                                                                                               \
            {}                                                                                                              \
                                                                                                                            \
            feature_result_property(typename base_type::result_type const & res)                                            \
                : detail::feature_result_property_impl< TAG, base_type, has_feature<                                        \
                      TAG, typename base_type::result_type                                                                  \
                  >::value>(res)                                                                                            \
            {}                                                                                                              \
    };

#endif
