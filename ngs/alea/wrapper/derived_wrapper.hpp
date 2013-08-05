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

#ifndef ALPS_NGS_ALEA_DERIVED_WRAPPER_HPP
#define ALPS_NGS_ALEA_DERIVED_WRAPPER_HPP

#include <alps/ngs/stacktrace.hpp>

#include <alps/ngs/alea/features.hpp>
#include <alps/ngs/alea/wrapper/base_wrapper.hpp>
#include <alps/ngs/alea/wrapper/result_type_wrapper.hpp> 

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

//this class holds the actual result
            template <typename Result, typename base_type> struct derived_result_wrapper_base: public base_type {
                typedef Result result_type;

                derived_result_wrapper_base(Result const & res): result_(res) {}

                result_type result_; // TODO: make this private!
            };

            template <typename Result> class derived_result_wrapper: public 
// TODO: generate form all_tags ...
                feature_result_property<tag::histogram,
                feature_result_property<tag::detail::tau,
                feature_result_property<tag::detail::converged,
                feature_result_property<tag::autocorrelation,
                feature_result_property<tag::log_binning,
                feature_result_property<tag::max_num_binning,
                feature_result_property<tag::fixed_size_binning,
                feature_result_property<tag::error,
                feature_result_property<tag::mean,

                derived_result_wrapper_base<
                    Result, result_type_result_wrapper<typename value_type<Result>::type>
                >
            > > > > > > > > > {
                //for nicer syntax
                typedef typename value_type<Result>::type value_type;
                typedef 
// TODO: generate form all_tags ...
                    feature_result_property<tag::histogram,
                    feature_result_property<tag::detail::tau,
                    feature_result_property<tag::detail::converged,
                    feature_result_property<tag::autocorrelation,
                    feature_result_property<tag::log_binning,
                    feature_result_property<tag::max_num_binning,
                    feature_result_property<tag::fixed_size_binning,
                    feature_result_property<tag::error,
                    feature_result_property<tag::mean,

                    derived_result_wrapper_base<
                        Result, detail::result_type_result_wrapper<value_type>
                    >
                > > > > > > > > > base_type;

                public:
                    using derived_result_wrapper_base<Result, result_type_result_wrapper<value_type> >::result_;

                    derived_result_wrapper(): base_type() {}

                    derived_result_wrapper(Result const & res): base_type(res) {}

                    detail::base_result_wrapper* clone() {return new derived_result_wrapper<Result>(result_);}

                    boost::uint64_t count() const {
                        return count_wrap(result_);
                    }

                    void print(std::ostream & out) {
                        out << result_;
                    }
            };

            //this class holds the actual accumulator
            template <typename Accum, typename base_type> struct derived_accumulator_wrapper_base: public base_type {
                typedef Accum accum_type;

                derived_accumulator_wrapper_base() {}

                derived_accumulator_wrapper_base(Accum const & acc): accum_(acc) {}

                accum_type accum_; // TODO: make this private!
            };
            
            //is needed for custom accumulators that don't have an operator()
            //they define their own add_value and the wrapper calls add value
            template<typename T, typename ValueType>
            void add_value(T & acc, ValueType const & val)
            {
                acc(val);
            }
            
            //the effective wrapper
            template <typename Accum> class derived_accumulator_wrapper: public 
// TODO: generate form all_tags ...
                feature_accumulator_property<tag::detail::weight,
                feature_accumulator_property<tag::histogram,
                feature_accumulator_property<tag::detail::tau,
                feature_accumulator_property<tag::detail::converged,
                feature_accumulator_property<tag::autocorrelation,
                feature_accumulator_property<tag::log_binning,
                feature_accumulator_property<tag::max_num_binning,
                feature_accumulator_property<tag::fixed_size_binning,
                feature_accumulator_property<tag::error,
                feature_accumulator_property<tag::mean,

                derived_accumulator_wrapper_base<
                    Accum, result_type_accumulator_wrapper<typename value_type<Accum>::type>
                >
            > > > > > > > > > > {
                //for nicer syntax
				typedef typename alps::accumulator::value_type<Accum>::type value_type;
                typedef typename weight_value_type<Accum>::type weight_value_type;
                typedef 
// TODO: generate form all_tags ...
                    feature_accumulator_property<tag::detail::weight,
                    feature_accumulator_property<tag::histogram,
                    feature_accumulator_property<tag::detail::tau,
                    feature_accumulator_property<tag::detail::converged,
                    feature_accumulator_property<tag::autocorrelation,
                    feature_accumulator_property<tag::log_binning,
                    feature_accumulator_property<tag::max_num_binning,
                    feature_accumulator_property<tag::fixed_size_binning,
                    feature_accumulator_property<tag::error,
                    feature_accumulator_property<tag::mean,

                    derived_accumulator_wrapper_base<
                        Accum, detail::result_type_accumulator_wrapper<value_type>
                    >
                > > > > > > > > > > base_type;

                template<int i> struct check_helper { typedef char type; };
                template<typename U> static char check_result(typename check_helper<sizeof(&U::result_type)>::type);
                template<typename U> static double check_result(...);

                public:
                    using derived_accumulator_wrapper_base<Accum, result_type_accumulator_wrapper<value_type> >::accum_;
                    
                    derived_accumulator_wrapper(): base_type() {}
                    
                    derived_accumulator_wrapper(Accum const & acc): base_type(acc) {}
                    
                    detail::base_accumulator_wrapper* clone() {return new derived_accumulator_wrapper<Accum>(accum_);}
                    
                    boost::uint64_t count() const {
                        return count_wrap(accum_);
                    }

                    void save(hdf5::archive & ar) const {
                        ar[""] = accum_;
                    }

                    void load(hdf5::archive & ar) {
                        ar[""] >> accum_;
                    }

                    void reset() {
                        reset_wrap(accum_);
                    }
                    
                    void print(std::ostream & out) {
                        out << accum_;
                    }

                    boost::shared_ptr<base_result_wrapper> result() const {
// TODO: fix in gcc-4.2
                        // return result_impl(check_helper<sizeof(check_result<Accum>(0))>());
                    }

#ifdef ALPS_HAVE_MPI
                private:
                    template<typename U, void (U::*)(boost::mpi::communicator const &, int) > struct check_collective_merge_helper {};
                    template<typename U> char check_collective_merge(check_collective_merge_helper<U, &U::collective_merge>*);
                    template<typename U> double check_collective_merge(...);
                public:

                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
// TODO: fix in gcc-4.2
//                        collective_merge_impl(comm, root, check_helper<sizeof(check_collective_merge<Accum>(0))>());
                    }
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
// TODO: fix in gcc-4.2
//                        collective_merge_impl(comm, root, check_helper<sizeof(check_collective_merge<Accum>(0))>());
                    }
                private:

                    void collective_merge_impl(
                          boost::mpi::communicator const & comm
                        , int root
                        , check_helper<sizeof(char)>
                    ) {
                        accum_.collective_merge(comm, root);
                    }

                    void collective_merge_impl(
                          boost::mpi::communicator const & comm
                        , int root
                        , check_helper<sizeof(char)>
                    ) const {
                        accum_.collective_merge(comm, root);
                    }

                    void collective_merge_impl(
                          boost::mpi::communicator const & comm
                        , int root
                        , check_helper<sizeof(double)>
                    ) const {
                        throw std::logic_error("The Accumulator has no collective_merge function" + ALPS_STACKTRACE);
                    }
#endif

                protected:

                    void add_value_wrap(void const * value, std::type_info const & vt_info) { //type-infusion
                        if( &vt_info != &typeid(value_type) &&
                        #ifdef BOOST_AUX_ANY_TYPE_ID_NAME
                            std::strcmp(vt_info.name(), typeid(value_type).name()) != 0
                        #else
                            vt_info != typeid(value_type)
                        #endif
                         )
                            throw std::runtime_error("wrong value type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
                        
                        //~ using alps::accumulator::detail::add_value;
                        
                        add_value(accum_, *static_cast<value_type const *>(value));
                    }
                    void add_value_wrap(void const * value, std::type_info const & vt_info
                                , void const * weight, std::type_info const & wvt_info) { //type-infusion with weight
                    //------------------- check value type -------------------
                        if( &vt_info != &typeid(value_type) &&
                        #ifdef BOOST_AUX_ANY_TYPE_ID_NAME
                            std::strcmp(vt_info.name(), typeid(value_type).name()) != 0
                        #else
                            vt_info != typeid(value_type)
                        #endif
                         )
                            throw std::runtime_error("wrong value type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
                    //------------------- check weight value type -------------------
                        std::cout << "typeid(weight_value_type): " << typeid(weight_value_type).name() << std::endl;
                        std::cout << "twvt_info.name(): " << wvt_info.name() << std::endl;
                        if( &wvt_info != &typeid(weight_value_type) &&
                        #ifdef BOOST_AUX_ANY_TYPE_ID_NAME
                            std::strcmp(wvt_info.name(), typeid(weight_value_type).name()) != 0
                        #else
                            wvt_info != typeid(weight_value_type)
                        #endif
                         )
                            throw std::runtime_error("wrong weight value type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
                        impl_picker<Accum, has_operator_for_weight<Accum>::value>().add_value(value, weight, accum_);
                        //~ accum_(*static_cast<value_type const *>(value), *static_cast<weight_value_type const *>(weight));
                    }

                private:

                    boost::shared_ptr<base_result_wrapper> result_impl(check_helper<sizeof(char)>) const {
                        return boost::shared_ptr<base_result_wrapper>(new derived_result_wrapper<typename Accum::result_type>(typename Accum::result_type(accum_)));
                    }

                    boost::shared_ptr<base_result_wrapper> result_impl(check_helper<sizeof(double)>) const {
                        throw std::logic_error("The Accumulator has no associated result_type" + ALPS_STACKTRACE);
                        return boost::shared_ptr<base_result_wrapper>((base_result_wrapper *)NULL);
                    }
                //------------------- find out it Accum has op()(vt, wvt) -------------------
                    template<typename T> 
                    struct has_operator_for_weight
                    {
                        template<typename TT>
                        struct has_type //checks if T has an type weight_tag_type (library accumulator)
                        {
                            template<typename U> static char check(typename U::weight_tag_type*);
                            template<typename U> static double check(...);
                            enum { value = (sizeof(char) == sizeof(check<TT>(0)))};
                        };
                        template<bool b, typename TT> //TT only needed since no explicid spec allowed
                        struct helper // in case is has the weight_tag_type it means that this is the effective type that has the operator we need
                        {
                            template<void(T::weight_tag_type::*)(int const &)> struct helper2
                            {
                                typedef char type;
                            };
                        };
                        template<typename TT> //TT only needed since no explicid spec allowed
                        struct helper<false, TT> //for custom accums that have the op not derived
                        {
                            template<void(T::*)(int const &)>
                            struct helper2
                            {
                                typedef char type;
                            };
                        };
                        
                        template<typename U> static char check(typename helper<has_type<T>::value, int>::template helper2<&U::operator()>::type);
                        template<typename U> static double check(...);
                        enum { value = (sizeof(char) == sizeof(check<T>(0))) };
                    };
                //------------------- impl picker for add_value(vt, wvt) -------------------
                    template<typename accum_type, bool b>
                    struct impl_picker
                    {
                        inline void add_value(void const * value, void const * weight, accum_type & acc)
                        {
                            std::cout << "type" << typeid(accum_type).name() << " \n" << has_operator_for_weight<Accum>::value << std::endl;
                            acc(*static_cast<value_type const *>(value), *static_cast<weight_value_type const *>(weight));
                        }
                    };
                    template<typename accum_type>
                    struct impl_picker<accum_type, false>
                    {
                        inline void add_value(void const * value, void const * weight, accum_type & acc)
                        {
                            std::cout << "type" << typeid(accum_type).name() << " \n" << has_operator_for_weight<Accum>::value << std::endl;
                            throw std::runtime_error(" no operator()(vt, wvt) in this accumulator (i.e. no weight-accumulator)" + ALPS_STACKTRACE);
                        }
                    };
            };
        }
    }
}
#endif
