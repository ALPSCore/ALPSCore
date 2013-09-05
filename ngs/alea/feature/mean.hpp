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

#ifndef ALPS_NGS_ALEA_DETAIL_MEAN_IMPLEMENTATION_HEADER
#define ALPS_NGS_ALEA_DETAIL_MEAN_IMPLEMENTATION_HEADER
 
#include <alps/ngs/short_print.hpp>
#include <alps/ngs/numeric/array.hpp>
#include <alps/ngs/numeric/detail.hpp>
#include <alps/ngs/numeric/vector.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/multi_array.hpp>

#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>

#include <alps/multi_array.hpp>
#include <alps/type_traits/element_type.hpp>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

namespace alps {
    namespace accumulator {       
        //=================== mean trait ===================
        namespace detail {
            template<unsigned n> struct static_array {
                char type[n];
            };
            
            template <typename T, int> struct mean_type_impl {
                typedef T type;
            };
         
            template <typename T> struct mean_type_impl<T, 2> {
                typedef double type;
            };
         
            template <typename T> struct mean_type_impl<T, 3> {
                typedef typename boost::is_same<T, T>::type false_type;
                BOOST_STATIC_ASSERT_MSG(!false_type::value, "mean_type trait failed");
            };
        }
        
        template <typename value_type> struct mean_type {
            private:
                typedef value_type T;
                static T t;
                static detail::static_array<1> test(T);
                static detail::static_array<2> test(double);
                static detail::static_array<3> test(...);
            public:
                typedef typename detail::mean_type_impl<T, sizeof(test((t+t)/double(1)))/sizeof(char)>::type type;
        };

        template<> struct mean_type<double> {
            public:
                typedef double type;
        };
        
        template<typename T> struct mean_type<std::vector<T> > {
            public:
                typedef std::vector<typename mean_type<T>::type > type;
        };
        
        template<typename T, std::size_t N> struct mean_type<boost::array<T, N> > {
            public:
                typedef boost::array<typename mean_type<T>::type, N> type;
        };
        
        template<typename T, std::size_t N> struct mean_type<boost::multi_array<T, N> > {
            public:
                typedef boost::multi_array<typename mean_type<T>::type, N> type;
        };

        //=================== mean implementation ===================
        namespace detail {
            //setting up the dependencies for tag::mean-Implementation isn't neccessary bc has none
            
            template<typename base_type> class AccumulatorImplementation<tag::mean, base_type> : public base_type {
                typedef typename base_type::value_type value_type_loc;
                typedef typename mean_type<value_type_loc>::type mean_type;
                typedef AccumulatorImplementation<tag::mean, base_type> ThisType;
                public:
                    AccumulatorImplementation<tag::mean, base_type>(ThisType const & arg): base_type(arg), sum_(arg.sum_) {}
                    
                    template<typename ArgumentPack> AccumulatorImplementation<tag::mean, base_type>(
                          ArgumentPack const & args
                        , typename boost::disable_if<boost::is_base_of<ThisType, ArgumentPack>, int>::type = 0
                    )
                        : base_type(args)
                        , sum_()
                    {}
                    
                    inline mean_type const mean() const {
                        using alps::ngs::numeric::operator/;
                        return mean_type(sum_) / (typename alps::hdf5::scalar_type<value_type_loc>::type)base_type::count();
                    }
            
                    inline void operator()(value_type_loc const & val)  {
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::detail::check_size;
                        base_type::operator()(val);
                        
                        check_size(sum_, val);
                        sum_ += val;
                    }
                    inline ThisType& operator<<(value_type_loc const & val)  {
                        (*this)(val);
                        return (*this);
                    }

                    template<typename Stream> inline void print(Stream & os) {
                        base_type::print(os);
                        os << "tag::mean: " << alps::short_print(mean()) << " " << std::endl;
                    }

                    void save(hdf5::archive & ar) const {
                        base_type::save(ar);
                        ar["mean/value"] = mean();
                    }

                    void load(hdf5::archive & ar) {
                        using alps::ngs::numeric::operator*;

                        base_type::load(ar);
                        value_type_loc mean;
                        ar["mean/value"] >> mean;
                        sum_ = mean * (typename alps::hdf5::scalar_type<value_type_loc>::type)base_type::count();
                    }

                    inline void reset() {
                        base_type::reset();
                        sum_ = value_type_loc();
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        base_type::collective_merge(comm, root);
                        if (comm.rank() == root)
                            base_type::reduce_if(comm, sum_, sum_, std::plus<typename alps::hdf5::scalar_type<value_type_loc>::type>(), root);
                        else
                            const_cast<ThisType const *>(this)->collective_merge(comm, root);
                    }                    
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        base_type::collective_merge(comm, root);
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                        else
                            base_type::reduce_if(comm, sum_, std::plus<typename alps::hdf5::scalar_type<value_type_loc>::type>(), root);
                    }
#endif

                protected:
                    value_type_loc sum_;
            };

            template<typename base_type> class ResultImplementation<tag::mean, base_type> : public base_type  {

                typedef typename mean_type<typename base_type::value_type>::type mean_type;

                public:

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : base_type(accum)
                        , mean_(accum.mean())
                    {}

                    inline mean_type const mean() const { 
                        return mean_;
                    }
// TODO: implement!
                protected:
                    mean_type mean_;
            };

        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(mean, tag::mean)

    }
}

#endif // ALPS_NGS_ALEA_DETAIL_MEAN_IMPLEMENTATION
