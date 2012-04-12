/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Mario Koenz <mkoenz@ethz.ch>                       *
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


#ifndef ALPS_NGS_ALEA_DETAIL_ACCUM_WRAPPER_MACRO_HEADER
#define ALPS_NGS_ALEA_DETAIL_ACCUM_WRAPPER_MACRO_HEADER

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/alea/detail/accum_prewrapper.hpp>

#include <boost/utility.hpp>
#include <boost/type_traits.hpp>

#include <typeinfo>
#include <sstream>
#include <stdexcept>



/*  this macro generates a set of function templates:
* 1. has_FCT<Accum>
*       Checks wether Accum has the method FCT or not
*       Result is retrived with has_FCT::value
*   
* 2. FCT(Accum)
*       Wrapps Accum.FCT into an extern function, if Accum.FCT exists
*   
* 3. detail::FCT_impl(Accum)
*       Calls FCT(Accum)
*   
* 4. detail::FCT_property_impl
*       Is the class, that contains the method FCT, which calls FCT_impl if Accum::FCT exists, otherwise throws an runtime-error 
*       if called. (now you see why FCT_impl is neccessary. Otherwise FCT_property_impl::FCT would call FCT and you get a 
*       recursion...)
*       Copy-ctor is needed to pass up the arguments
* 
* 5. FCT_property
*       Derives from the right detail::FCT_property_impl (with/without Accum::FCT) and implements the copy-ctor to pass 
*       up the argumnets
*   
*/
namespace alps
{
    namespace alea
    {
        #define IMPLEMENT_FUNCTION(FCT) \
        \
        /* = = = = = = = = = = I N F O   T R A I T = = = = = = = = = = */\
            template<typename Accum> \
            struct has_ ## FCT \
            { \
                template<int i> struct helper { \
                    typedef char type; \
                }; \
                template<typename U> static char check(typename helper<sizeof(&U::FCT)>::type); \
                template<typename U> static double check(...); \
                \
                enum \
                { \
                    value = (sizeof(char) == sizeof(check<Accum>(0))) \
                }; \
            };\
            \
        /* = = = = = = = = = = F C T   V I A   M E M B E R = = = = = = = = = = */\
        \
        template <typename Accum>\
        inline typename FCT ## _type<typename value_type<Accum>::type>::type FCT(Accum const & arg)\
        {\
            return arg.FCT();\
        }\
        namespace detail \
        { \
        /* = = = = = = = = = = A V O I D S   N A M E C O N F L I C T S = = = = = = = = = = */\
        \
            template<typename Accum>\
            inline typename FCT ## _type<typename value_type<Accum>::type >::type FCT ## _impl(Accum const & arg)\
            {\
                return FCT(arg);\
            }\
            \
            \
        /* = = = = = = = = = = P R O P E R T Y   I M P L   W I T H   F C T = = = = = = = = = = */\
        \
            template <typename base, bool> \
            class FCT ## _property_impl: public base \
            { \
            public:\
                FCT ## _property_impl() {}\
                FCT ## _property_impl(typename base::accum_type const & arg): base(arg) {}\
                \
                typename FCT ## _type<typename value_type<typename base::accum_type>::type >::type FCT() const \
                { \
                    return FCT ## _impl(base::accum_); \
                } \
            }; \
        /* = = = = = = = = = = P R O P E R T Y   I M P L   W I T H O U T   F C T = = = = = = = = = = */\
            template <typename base> \
            class FCT ## _property_impl<base, false>: public base \
            { \
            public:\
                FCT ## _property_impl() {}\
                FCT ## _property_impl(typename base::accum_type const & arg): base(arg) {}\
                \
                typename FCT ## _type<typename value_type<typename base::accum_type>::type >::type FCT() const \
                { \
                    std::stringstream out; \
                    out << typeid(typename base::accum_type).name(); \
                    out << " has no ";\
                    out << #FCT;\
                    out << "-method"; \
                    boost::throw_exception(std::runtime_error(out.str() + ALPS_STACKTRACE)); \
                    return typename FCT ## _type<typename value_type<typename base::accum_type>::type >::type(0); \
                } \
            };\
        } /*end namespace detail*/\
        \
        \
        /* = = = = = = D E R I V E   F R O M   T H E   R I G H T   F C T   I M P  L  = = = = = = */\
        template <typename base> \
        class FCT ## _property: public detail::FCT ## _property_impl< \
                                                              base \
                                                            , has_ ## FCT<typename base::accum_type>::value \
                                                            > \
        {\
        public:\
            FCT ## _property(typename base::accum_type const & acc): detail::FCT ## _property_impl< \
                                                              base \
                                                            , has_ ## FCT<typename base::accum_type>::value \
                                                            >(acc) \
                                                            {}\
        };

    }//end alea namespace 
}//end alps namespace
#endif //ALPS_NGS_ALEA_DETAIL_ACCUM_WRAPPER_MACRO_HEADER
