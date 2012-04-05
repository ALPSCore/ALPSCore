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


#ifndef ALPS_NGS_ALEA_DETAIL_ACCUM_WRAPPER_HEADER
#define ALPS_NGS_ALEA_DETAIL_ACCUM_WRAPPER_HEADER

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/alea/detail/accum_prewrapper.hpp>
#include <alps/ngs/alea/measurement_fwd.hpp>
#include <alps/ngs/alea/extern_function.hpp>

#include <alps/ngs/alea/detail/properties.hpp>

#include <boost/cstdint.hpp>

#include <typeinfo> //used in add_value
#include <stdexcept>

namespace alps
{
    namespace alea
    {
        namespace detail
        {
        //= = = = = = = = = = = = = = = = = = W R A P P E R   B A S E = = = = = = = = = = = = = = =
        //declaration because needed in base_wrapper

            template <typename value_type> 
            class result_type_wrapper;
            
            //base of result_type_wrapper. Defines the usable interface
            class base_wrapper {
                public:
                    base_wrapper() {}
                    
                    template<typename value_type>
                    void operator<<(value_type& value) 
                    {
                        add_value(&value, typeid(value_type));
                    }
                    
                    template<typename value_type>
                    result_type_wrapper<value_type> &get() 
                    {
                        return dynamic_cast<result_type_wrapper<value_type>& >(*this);
                    }
                    
                    virtual base_wrapper* clone() = 0;  //needed for the copy-ctor
                    virtual void print(std::ostream & out) = 0;
                    
                protected:
                    virtual void add_value(const void* value, const std::type_info& t_info) = 0; //for operator<<
            };

        //= = = = = = = = = = = = = = = = = = R E S U L T   T Y P E   W R A P P E R = = = = = = = = = = = = = = =
        //returns mean and other data that needs the type and therefore can't be implemented in base_wrapper

            template <typename value_type> 
            class result_type_wrapper: public base_wrapper 
            {
                public:
                    virtual typename mean_type<value_type>::type mean() const= 0;
                    virtual typename error_type<value_type>::type error() const = 0;
                    virtual boost::int64_t count() const = 0;
                    virtual typename fix_size_bin_type<value_type>::type fix_size_bin() const = 0;
                    virtual typename max_num_bin_type<value_type>::type max_num_bin() const = 0;
                    virtual typename log_bin_type<value_type>::type log_bin() const = 0;
                    virtual typename autocorr_type<value_type>::type autocorr() const = 0;
            };
        //= = = = = = = = = = = = = = = = = = A C C U M U L A T O R   W R A P P E R = = = = = = = = = = = = = = =
        //the effective wrapper

            template <typename Accum> 
            class accumulator_wrapper: public   autocorr_property<
                                                 log_bin_property<
                                                  max_num_bin_property<
                                                   fix_size_bin_property<
                                                    error_property<
                                                     mean_property<
                                                      accumulator_prewrapper<
                                                                            Accum
                                                                          , detail::result_type_wrapper<
                                                                                                    typename value_type<Accum>::type
                                                                                                       >
                                                                            >
                                                     >
                                                    >
                                                   >
                                                  >
                                                 >
                                                >
            {
                    //for nicer syntax
                    typedef typename value_type<Accum>::type value_type;
                    typedef autocorr_property<
                             log_bin_property<
                              max_num_bin_property<
                               fix_size_bin_property<
                                error_property<
                                 mean_property<
                                  accumulator_prewrapper<
                                                        Accum
                                                      , result_type_wrapper<value_type> 
                                                        > 
                                 > 
                                >
                               >
                              >
                             >
                            > base;
                    
                    using accumulator_prewrapper<Accum, result_type_wrapper<value_type> >::accum_;
                    
                public:
                    accumulator_wrapper(): base() {}
                    
                    accumulator_wrapper(Accum const & acc): base(acc) {}
                    
                    detail::base_wrapper* clone() {return new accumulator_wrapper<Accum>(accum_);}
                    
                    boost::int64_t count() const
                    {
                        return detail::count_wrap(accum_);
                    }
                    
                    void print(std::ostream & out) {out << accum_;}
                    
                    friend Accum& measurement::extract<Accum>();
                
                protected:
                    void add_value(const void* value, const std::type_info& info) //type-infusion
                    {
                        if( &info != &typeid(value_type) &&
                        #ifdef BOOST_AUX_ANY_TYPE_ID_NAME
                            std::strcmp(info.name(), typeid(value_type).name()) != 0
                        #else
                            info != typeid(value_type)
                        #endif
                         )
                            std::runtime_error("wrong type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
                            //~ std::runtime_error("wrong type added in accumulator_wrapper::add_value"  + alps::ngs::stacktrace());
                        accum_ << *static_cast<const value_type*>(value);
                    }
            };
        }//end of detail namespace----------------------------------------------
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_ACCUM_WRAPPER_HEADER
