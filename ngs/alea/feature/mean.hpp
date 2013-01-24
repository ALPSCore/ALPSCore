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


#ifndef ALPS_NGS_ALEA_DETAIL_MEAN_IMPLEMENTATION_HEADER
#define ALPS_NGS_ALEA_DETAIL_MEAN_IMPLEMENTATION_HEADER

#include <alps/ngs/alea/accumulator/accumulator_impl.hpp>
#include <alps/ngs/short_print.hpp>

#include <alps/ngs/alea/features.hpp>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

namespace alps
{
    namespace accumulator
    {
        //=================== mean trait ===================
        namespace detail
        {
            template<unsigned n> struct static_array
            {
                char type[n];
            };
            
            template <typename T, int>
            struct mean_type_impl
            {
                typedef T type;
            };
         
            template <typename T>
            struct mean_type_impl<T, 2>
            {
                typedef double type;
            };
         
            template <typename T>
            struct mean_type_impl<T, 3>
            {
                typedef typename boost::is_same<T, T>::type false_type;
                BOOST_STATIC_ASSERT_MSG(!false_type::value, "mean_type trait failed");
            };
        }
        
        //~ using namespace alps::numeric;
        
        template <typename value_type>
        struct mean_type
        {
            private:
                typedef value_type T;
                static T t;
                static detail::static_array<1> test(T);
                static detail::static_array<2> test(double);
                static detail::static_array<3> test(...);
            public:
                typedef typename detail::mean_type_impl<T, sizeof(test((t+t)/double(1)))/sizeof(char)>::type type;
        };

        template<>
        struct mean_type<double>
        {
            public:
                typedef double type;
        };
        
        template<typename T>
        struct mean_type<std::vector<T> >
        {
            public:
                typedef std::vector<typename mean_type<T>::type > type;
        };
        
        template<typename T, std::size_t N>
        struct mean_type<boost::array<T, N> >
        {
            public:
                typedef boost::array<typename mean_type<T>::type, N> type;
        };
        
        template<typename T, std::size_t N>
        struct mean_type<boost::multi_array<T, N> >
        {
            public:
                typedef boost::multi_array<typename mean_type<T>::type, N> type;
        };
        //=================== mean implementation ===================
        namespace detail
        {
            //setting up the dependencies for tag::mean-Implementation isn't neccessary bc has none
            
            template<typename base_type> 
            class Implementation<tag::mean, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
                typedef typename mean_type<value_type_loc>::type mean_type;
                typedef Implementation<tag::mean, base_type> ThisType;
                public:
                    Implementation<tag::mean, base_type>(ThisType const & arg): base_type(arg), sum_(arg.sum_) {}
                    
                    template<typename ArgumentPack>
                    Implementation<tag::mean, base_type>(ArgumentPack const & args, typename boost::disable_if<
                                                                                          boost::is_base_of<ThisType, ArgumentPack>
                                                                                        , int
                                                                                        >::type = 0
                                        ): base_type(args)
                                         , sum_()
                    {}
                    
                    inline mean_type const  mean() const 
                    { 
                        using alps::ngs::numeric::operator/;
                        
                        return mean_type(sum_) / base_type::count();
                    }
            
                    inline ThisType& operator <<(value_type_loc val) 
                    {
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::detail::check_size;
                        base_type::operator <<(val);
                        
                        check_size(sum_, val);
                        sum_ += val;
                        return *this;
                    }
            
                    template<typename Stream> 
                    inline void print(Stream & os) 
                    {
                        base_type::print(os);
                        os << "tag::mean: " << alps::short_print(mean()) << " " << std::endl;
                    }
                    inline void reset()
                    {
                        base_type::reset();
                        sum_ = value_type_loc();
                    }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {
                        base_type::collective_merge(comm, root);
                        // TODO: make alps::mpi::reduce
                        // TODO: use std::plus<alps::element_type<...> >
                        /*
                        if (comm.rank() == root)
                            boost::mpi::reduce(comm, sum_, sum_, std::plus<value_type_loc>(), root);
                        else
                            boost::mpi::reduce(comm, sum_, std::plus<value_type_loc>(), root);
                        */
                    }
#endif

                protected:
                    value_type_loc sum_;
            };
        } // end namespace detail
    }//end accumulator namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_MEAN_IMPLEMENTATION
