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

#ifndef ALPS_NGS_ALEA_BASE_WRAPPER_HPP
#define ALPS_NGS_ALEA_BASE_WRAPPER_HPP

#include <typeinfo> //used in add_value
#include <stdexcept>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

#include <boost/cstdint.hpp>
#include <alps/ngs/alea/accumulator/arguments.hpp>

namespace alps {
    namespace accumulator {
        namespace detail {
            
            //this one is needed, bc of name-collision in accum_wrapper
            template<typename Accum> boost::uint64_t count_wrap(Accum const & arg) {
                return count(arg);
            }

            template<typename Accum> void reset_wrap(Accum & arg) {
                return reset(arg);
            }
            
            //declaration because needed in base_accumulator_wrapper
            template <typename value_type>  class result_type_result_wrapper;
            
            //base_type of result_type_result_wrapper. Defines the usable interface
            class base_result_wrapper {
                public:
                    base_result_wrapper() {}
                    virtual ~base_result_wrapper() {}

                    template<typename value_type> inline result_type_result_wrapper<value_type> &get()  {
                        return dynamic_cast<result_type_result_wrapper<value_type>& >(*this);
                    }

                    virtual boost::uint64_t count() const = 0;
                    virtual base_result_wrapper* clone() = 0;  //needed for the copy-ctor
                    virtual void print(std::ostream & out) = 0;
            };

            //declaration because needed in base_accumulator_wrapper
            template <typename value_type>  class result_type_accumulator_wrapper;
            
            //base_type of result_type_accumulator_wrapper. Defines the usable interface
            class base_accumulator_wrapper {
                public:
                    base_accumulator_wrapper() {}
                    virtual ~base_accumulator_wrapper() {}
                    
                    template<typename value_type> inline void operator()(value_type& value) {
                        add_value_wrap(&value, typeid(value_type));
                    }
                    template<typename value_type> inline void operator<<(value_type& value) {
                        (*this)(value);
                    }
                     //------------------- input with weight-------------------
                    template<typename value_type, typename weight_value_type>
                    inline void operator()(value_type const & value
                                          , weight_value_type const & weight
                                          , typename boost::disable_if<
                                                                        boost::is_base_of<boost::parameter::aux::tagged_argument_base
                                                                                        , weight_value_type
                                                                                         >
                                                                      , int
                                                                      >::type = 0) 
                    {
                        add_value_wrap(&value, typeid(value_type), &weight, typeid(weight_value_type));
                    }
                    template<typename value_type, typename ArgumentPack>
                    inline void operator()(value_type const & value
                                         , ArgumentPack const & argpac
                                         , typename boost::enable_if<
                                                                    boost::is_base_of<boost::parameter::aux::tagged_argument_base
                                                                                    , ArgumentPack
                                                                                     >
                                                                  , int
                                                                    >::type = 0) 
                    {
                        add_value_wrap(&value, typeid(value_type), &(argpac[Weight]), typeid(argpac[Weight]));
                    }
                    
                    template<typename value_type> inline result_type_accumulator_wrapper<value_type> &get() {
                        return dynamic_cast<result_type_accumulator_wrapper<value_type>& >(*this);
                    }
                    
                    virtual boost::uint64_t count() const = 0;
                    virtual void save(hdf5::archive & ar) const = 0;
                    virtual void load(hdf5::archive & ar) = 0;
                    virtual void reset() = 0;
                    virtual base_accumulator_wrapper* clone() = 0;  //needed for the copy-ctor
                    virtual void print(std::ostream & out) = 0;
                    virtual boost::shared_ptr<base_result_wrapper> result() const = 0;

#ifdef ALPS_HAVE_MPI
                    virtual void collective_merge(boost::mpi::communicator const & comm, int root) = 0;
#endif

                protected:
                    virtual void add_value_wrap(void const * value, std::type_info const & vt_info) = 0; //for operator()
                    virtual void add_value_wrap(void const * value, std::type_info const & vt_info, void const * weight, std::type_info const & wvt_info) = 0; //for operator() with weight
            };
        }
    }
}
#endif
