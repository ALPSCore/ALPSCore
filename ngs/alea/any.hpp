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

#ifndef ALPS_NGS_ALEA_ANY_HPP
#define ALPS_NGS_ALEA_ANY_HPP

#include <alps/ngs/stacktrace.hpp>

#include <boost/utility.hpp>
#include <boost/type_traits.hpp>
#include <stdexcept>
#include <sstream>

namespace alps
{
    namespace alea
    {
        namespace detail
        {
        template<typename T> struct type {
            static void value() {}
        };
        
        template<void(* T)(), void(* U)()> 
        struct is_same_impl: public boost::false_type {};
        
        template<typename T, typename U> 
        struct is_same: public is_same_impl<&type<T>::value, &type<U>::value> {};
        
        template<typename T>
        struct is_same<T, T>: public boost::true_type {};
        
        template<typename T>
        class make_any {
            public:
                make_any(T & arg): data(arg) {}
                
                void * operator()() {
                    return &data;
                }
            private:
                T & data;
        };
        
        class weak_type_ptr {
            public:
                template<typename T>
                weak_type_ptr(T & arg): fct(&type<T>::value), data(&arg) {}
                
                template<typename T> T & cast() {
                    if(fct != &type<T>::value) {
                        std::stringstream out;
                        out << "bad cast in alps::alea::detail weak_type_ptr.cast<type>()";
                        boost::throw_exception(std::runtime_error(out.str() + ALPS_STACKTRACE));
                    }
                    return *static_cast<T*>(data);
                }

            private:
                void (*fct)();
                void * data;
        };
        
        
        template<typename T> class make_data
        {
            public:
                make_data(T const & arg): data(arg) {}

                weak_type_ptr operator()() {
                    return weak_type_ptr(data);
                }
                
                T & get() {
                    return data;
                }
            
            private:
                T data;
        };
        
        //~ template<typename T> struct make_any {
                        //~ make_ref_ptr(T & arg) : data(arg) {}
                        //~ void * operator()(type_info) {
        //~ check type_it ..
        //~ return &data; }
                        //~ T & data;
                //~ };

        //~ ctor
        //~ boost::function<void *(type_info)> ptr(make_any(mytype));

        //~ cast
        //~ *static_cast<mytype *>(ptr(typeid(mytype)));


        } // end namespace detail
    }//end accumulator namespace 
}//end alps namespace
#endif //ALPS_NGS_ALEA_ANY_HPP
