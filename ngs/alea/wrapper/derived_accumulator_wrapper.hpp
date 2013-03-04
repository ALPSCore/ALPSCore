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

#ifndef ALPS_NGS_ALEA_ACCUMULATOR_DERIVED_WRAPPER_HEADER
#define ALPS_NGS_ALEA_ACCUMULATOR_DERIVED_WRAPPER_HEADER

#include <alps/ngs/stacktrace.hpp>

#include <alps/ngs/alea/features.hpp>
#include <alps/ngs/alea/accumulator/properties.hpp>
#include <alps/ngs/alea/wrapper/base_accumulator_wrapper.hpp>
#include <alps/ngs/alea/wrapper/result_type_accumulator_wrapper.hpp> 

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>

#include <typeinfo> //used in add_value
#include <stdexcept>

namespace alps {
    namespace accumulator {
        namespace detail {

        //= = = = = = = = = = = = = = = = = = P R E   W R A P P E R = = = = = = = = = = = = = = =
        //this class holds the actual accumulator
            template <typename Accum, typename result_type_base_type> class accumulator_prewrapper: public result_type_base_type {
                public:
                    typedef Accum accum_type;

                    accumulator_prewrapper(Accum const & acc): accum_(acc)
                    {}

                    Accum accum_; // TODO: make this private!
            };

        //= = = = = = = = = = = = = = = = = = D E R I V E D   W R A P P E R = = = = = = = = = = = = = = =
        //the effective wrapper

            // TODO: move XXX_property to the acording feature
            template <typename Accum> 
            class derived_accumulator_wrapper: public histogram_property <tau_property <converged_property<autocorr_property<
                log_bin_property<max_num_bin_property<fixed_size_bin_property<error_property<mean_property<accumulator_prewrapper<
                    Accum, result_type_accumulator_wrapper<typename value_type<Accum>::type>
                > > > > > > 
            > > > > {
                //for nicer syntax
                typedef typename value_type<Accum>::type value_type;
                typedef histogram_property <
                    tau_property <converged_property<autocorr_property<log_bin_property<max_num_bin_property<
                        fixed_size_bin_property<error_property<mean_property<accumulator_prewrapper<
                            Accum, detail::result_type_accumulator_wrapper<value_type>
                    > > > > 
                > > > > > > base_type;

                public:
                    using accumulator_prewrapper<Accum, result_type_accumulator_wrapper<value_type> >::accum_;
                    
                    derived_accumulator_wrapper(): base_type() {}
                    
                    derived_accumulator_wrapper(Accum const & acc): base_type(acc) {}
                    
                    inline detail::base_accumulator_wrapper* clone() {return new derived_accumulator_wrapper<Accum>(accum_);}
                    
                    inline boost::uint64_t count() const {
                        return count_wrap(accum_);
                    }

                    void reset() {
                        reset_wrap(accum_);
                    }
                    
                    inline void print(std::ostream & out) {
                        out << accum_;
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        collective_merge_impl(comm, root, collective_merge_helper<sizeof(check<Accum>(0))>());
                    }
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        collective_merge_impl(comm, root, collective_merge_helper<sizeof(check<Accum>(0))>());
                    }
                private:
                    template<int i> struct collective_merge_helper { typedef char type; };
                    template<typename U> static char check(typename collective_merge_helper<sizeof(&U::collective_merge)>::type);
                    template<typename U> static double check(...);

                    void collective_merge_impl(
                          boost::mpi::communicator const & comm
                        , int root
                        , collective_merge_helper<sizeof(char)>
                    ) {
                        accum_.collective_merge(comm, root);
                    }

                    void collective_merge_impl(
                          boost::mpi::communicator const & comm
                        , int root
                        , collective_merge_helper<sizeof(char)>
                    ) const {
                        accum_.collective_merge(comm, root);
                    }

                    void collective_merge_impl(
                          boost::mpi::communicator const & comm
                        , int root
                        , collective_merge_helper<sizeof(double)>
                    ) const {
                        throw std::logic_error("The Accumulator has no collective_merge function" + ALPS_STACKTRACE);
                    }
#endif

                protected:
                    inline void add_value(void const * value, std::type_info const & info) //type-infusion
                    {
                        if( &info != &typeid(value_type) &&
                        #ifdef BOOST_AUX_ANY_TYPE_ID_NAME
                            std::strcmp(info.name(), typeid(value_type).name()) != 0
                        #else
                            info != typeid(value_type)
                        #endif
                         )
                            throw std::runtime_error("wrong type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
                        accum_ << *static_cast<value_type const *>(value);
                    }
            };
        }
    }
}
#endif
