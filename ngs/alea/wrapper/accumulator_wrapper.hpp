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


#ifndef ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HEADER
#define ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HEADER

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/alea/wrapper/extern_function.hpp>
#include <alps/ngs/alea/wrapper/accumulator_wrapper_fwd.hpp>

#include <alps/ngs/alea/accumulator/properties.hpp>

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

        //= = = = = = = = = = = = = = = = = = W R A P P E R   B A S E = = = = = = = = = = = = = = =
        //declaration because needed in base_accumulator_wrapper

            template <typename value_type> 
            class result_type_accumulator_wrapper;
            
            //base_type of result_type_accumulator_wrapper. Defines the usable interface
            class base_accumulator_wrapper {
                public:
                    base_accumulator_wrapper() {}
                    virtual ~base_accumulator_wrapper() {}
                    
                    template<typename value_type>
                    inline void operator<<(value_type& value) 
                    {
                        add_value(&value, typeid(value_type));
                    }
                    
                    template<typename value_type>
                    inline result_type_accumulator_wrapper<value_type> &get() 
                    {
                        return dynamic_cast<result_type_accumulator_wrapper<value_type>& >(*this);
                    }
                    
                    virtual boost::uint64_t count() const = 0;
                    virtual void reset() = 0;
                    virtual base_accumulator_wrapper* clone() = 0;  //needed for the copy-ctor
                    virtual void print(std::ostream & out) = 0;

#ifdef ALPS_HAVE_MPI
                    virtual void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) = 0;
#endif

                protected:
                    virtual void add_value(const void* value, const std::type_info& t_info) = 0; //for operator<<
            };

        //= = = = = = = = = = = = = = = = = = R E S U L T   T Y P E   W R A P P E R = = = = = = = = = = = = = = =
        //returns mean and other data that needs the type and therefore can't be implemented in base_accumulator_wrapper

            template <typename ValueType>
            class result_type_accumulator_wrapper: public base_accumulator_wrapper 
            {
                public:
                    typedef ValueType value_type;
                    virtual ~result_type_accumulator_wrapper() {}
                    virtual typename mean_type<value_type>::type mean() const = 0;
                    virtual bool has_mean() const = 0;
                    virtual typename error_type<value_type>::type error() const = 0;
                    virtual bool has_error() const = 0;
                    virtual typename fixed_size_bin_type<value_type>::type fixed_size_bin() const = 0;
                    virtual bool has_fixed_size_bin() const = 0;
                    virtual typename max_num_bin_type<value_type>::type max_num_bin() const = 0;
                    virtual bool has_max_num_bin() const = 0;
                    virtual typename log_bin_type<value_type>::type log_bin() const = 0;
                    virtual bool has_log_bin() const = 0;
                    virtual typename autocorr_type<value_type>::type autocorr() const = 0;
                    virtual bool has_autocorr() const = 0;
                    virtual typename converged_type<value_type>::type converged() const = 0;
                    virtual bool has_converged() const = 0;
                    virtual typename tau_type<value_type>::type tau() const = 0;
                    virtual bool has_tau() const = 0;
                    virtual typename histogram_type<value_type>::type histogram() const = 0;
                    virtual bool has_histogram() const = 0;
            };
        //= = = = = = = = = = = = = = = = = = D E R I V E D   W R A P P E R = = = = = = = = = = = = = = =
        //the effective wrapper

            template <typename Accum> 
            class derived_accumulator_wrapper: public histogram_property <
                tau_property <
                    converged_property<
                        autocorr_property<
                            log_bin_property<
                                max_num_bin_property<
                                    fixed_size_bin_property<
                                        error_property<
                                            mean_property<
                                                accumulator_prewrapper<
                                                    Accum, detail::result_type_accumulator_wrapper<typename value_type<Accum>::type>
                                                >
                                            >
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
                typedef histogram_property<
                    tau_property <
                        converged_property<
                            autocorr_property<
                                log_bin_property<
                                    max_num_bin_property<
                                        fixed_size_bin_property<
                                            error_property<
                                                mean_property<
                                                    accumulator_prewrapper<Accum, result_type_accumulator_wrapper<value_type> > 
                                                > 
                                            > 
                                        > 
                                    > 
                                >
                            >
                        >
                    >
                > base_type;
                    
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
                    
                    inline void print(std::ostream & out) {out << accum_;}
                    
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

        //= = = = = = = = = = = = = = = = = = A C C U M U L A T O R   W R A P P E R = = = = = = = = = = = = = = =
        //class that holds the base_accumulator_wrapper pointer
            template<typename T> 
            accumulator_wrapper::accumulator_wrapper(T arg): base_(new derived_accumulator_wrapper<T>(arg))
            {}

            template<typename T>
            accumulator_wrapper& accumulator_wrapper::operator<<(const T& value) 
            {
                (*base_) << value; return *this;
            }

            template<typename T>
            result_type_accumulator_wrapper<T> &accumulator_wrapper::get() const
            {
                return base_->get<T>();
            }
            
            template <typename T>
            T & accumulator_wrapper::extract() const
            {
                return (dynamic_cast<detail::derived_accumulator_wrapper<T>& >(*base_)).accum_;
            }

            inline boost::uint64_t accumulator_wrapper::count() const
            {
                return base_->count();
            }
            
            inline void accumulator_wrapper::reset()
            {
                base_->reset();
            }

#ifdef ALPS_HAVE_MPI
            inline void accumulator_wrapper::collective_merge(
                  boost::mpi::communicator const & comm
                , int root
            ) {
                base_->collective_merge(comm, root);
            }
            inline void accumulator_wrapper::collective_merge(
                  boost::mpi::communicator const & comm
                , int root
            ) const {
                base_->collective_merge(comm, root);
            }
#endif

            inline accumulator_wrapper::accumulator_wrapper(accumulator_wrapper const & arg): base_(arg.base_->clone()) 
            {}
                
            inline std::ostream& operator<<(std::ostream &out, const accumulator_wrapper& m)
            {
                m.base_->print(out);
                return out;
            }
        }//end detail namespace 
    }//end accumulator namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_ACCUMULATOR_WRAPPER_HEADER
