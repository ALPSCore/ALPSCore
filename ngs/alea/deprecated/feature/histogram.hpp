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
 
#ifndef ALPS_NGS_ALEA_DETAIL_HISTOGRAM_IMPLEMENTATION_HEADER
#define ALPS_NGS_ALEA_DETAIL_HISTOGRAM_IMPLEMENTATION_HEADER

#include <alps/ngs/alea/feature/mean.hpp>
#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>

#include <alps/ngs/stacktrace.hpp>

#include <boost/cstdint.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits.hpp>

#include <iostream>
#include <vector>
#include <numeric>  //for accumulate
#include <utility> //for pair

namespace alps {
    namespace accumulator {
        //=================== histogram proxy ===================
        namespace detail {
        //= = = = = = = = = = = H I S T O G R A M   P R O X Y = = = = = = = = = = = = =
            template <typename Hist>
            class histogram_old_proxy
            {
                typedef typename Hist::value_type value_type;
                typedef typename Hist::weight_type weight_type;
                
                public:
                    //ctor with parameter
                    histogram_old_proxy<Hist>(Hist & arg, value_type const pos): hist_ref_(arg), pos_(pos) 
                    {}
                
                    histogram_old_proxy<Hist>(histogram_old_proxy const & arg): hist_ref_(arg.hist_ref_), pos_(arg.pos_) 
                    {}
                    
                    //add value via += operator
                    void operator+=(weight_type const & arg) 
                    {
                        hist_ref_ << std::pair<value_type, weight_type>(pos_, arg);
                    }
                
                    //add value via ++pre operator
                    histogram_old_proxy<Hist> & operator++()
                    {
                        hist_ref_ << std::pair<value_type, weight_type>(pos_, weight_type(1));
                        return *this;
                    }
                    
                    //add value by post++
                    histogram_old_proxy<Hist> operator++(int) 
                    {
                        hist_ref_ << std::pair<value_type, weight_type>(pos_, weight_type(1));
                        return histogram_old_proxy<Hist>(*this);
                    }
                    
                    //caster to get value back
                    operator weight_type() const
                    {
                        return const_cast<Hist const &>(hist_ref_)[pos_];
                    }
                    
                    //print the private value
                    void print(std::ostream & os) const
                    {
                        os << const_cast<Hist const &>(hist_ref_)[pos_];
                    }

                private:
                    Hist & hist_ref_;
                    value_type const pos_;
            };

            template <typename T> std::ostream& operator<<(std::ostream& out,  const histogram_old_proxy<T>& d) {
                d.print(out);
                return out;
            }
        } // end namespace detail

        //= = = = = = = = = = = H I S T O G R A M   = = = = = = = = = = = = = = = = = =

        template <typename T, typename U = unsigned int>
        class histogram_old
        {
            typedef typename std::vector<U>::size_type size_type;

            public:
                typedef T value_type;
                typedef U weight_type;
                
                //TODO: check int vs double behavior
                template<typename V>
                histogram_old(V start, V end, size_type size, typename boost::enable_if<boost::is_integral<V>, int>::type = 0)
                                                                            : count_()
                                                                            , start_(start)
                                                                            , num_break_((end-start)/(size-1))
                                                                            , size_(size)
                                                                            , data_(size_, weight_type()) {}
                template<typename V>
                histogram_old(V start, V end, size_type size, typename boost::enable_if<boost::is_floating_point<V>, int>::type = 0)
                                                                            : count_()
                                                                            , start_(start)
                                                                            , num_break_((end-start)/(size))
                                                                            , size_(size)
                                                                            , data_(size_, weight_type()) {}
                
                //copy ctor
                histogram_old(histogram_old const & arg): count_(arg.count_)
                                                , start_(arg.start_)
                                                , num_break_(arg.num_break_)
                                                , size_(arg.size_)
                                                , data_(arg.data_) {}
                
                //insert a value via stream-operator
                void operator()(value_type const & val)
                {
                    using namespace alps::ngs::numeric;
                    data_[get_index(val)] += weight_type(1);
                    count_ += weight_type(1);
                }
                inline void operator<<(value_type const & val)
                {
                    (*this)(val);
                }

                //insert a pair via stream operator
                void operator()(std::pair<value_type, weight_type> p)
                {
                    using namespace alps::ngs::numeric;
                    data_[get_index(p.first)] += p.second;
                    count_ += p.second;
                }
                inline void operator<<(std::pair<value_type, weight_type> p)
                {
                    return (*this)(p);
                }
                
                //get the proxy
                detail::histogram_old_proxy<histogram_old<T, U> > operator[](value_type const & arg)
                {
                    return detail::histogram_old_proxy<histogram_old<T, U> >(*this, arg);
                }

                //const version
                weight_type operator[](value_type const & arg) const
                {
                    return data_[get_index(arg)];
                }

                //calculate the mean
                typename mean_type<weight_type>::type mean() const 
                {
                    typename mean_type<weight_type>::type res(0);
                    
                    using namespace alps::ngs::numeric;
                    
                    for(value_type i = start_; i < start_ + value_type(size_)*num_break_; i += num_break_)
                        res += i * data_[get_index(i)];
                    return res / count();
                }

                //return the count
                weight_type count() const
                {
                    return count_;
                }

            private:
                //get from user-index to impl-index for std::vector
                size_type get_index(value_type const & arg) const
                {
                    using namespace alps::ngs::numeric;
                    using std::floor;
                    if(floor((arg - start_) / num_break_) == double(size_))
                        return size_ - 1;
                    return (arg - start_) / num_break_;
                }
                
            private:
                weight_type count_;
                value_type start_;
                value_type num_break_;
                size_type size_;
                std::vector<weight_type> data_;
        };

        template <typename T, typename U>
        std::ostream& operator<<(std::ostream& out,  const histogram_old<T, U>& d)
        {
            out << "histogram_old: ";
            out << "mean: ";
            out << d.mean();
            return out;
        }
        //=================== histogram trait ===================
        template <typename T> struct histogram_type {
            typedef double type;
        };
        //=================== histogram implementation ===================
        namespace detail {

        //set up the dependencies for the tag::histogram-Implementation
            template<> 
            struct Dependencies<tag::histogram> 
            {
                typedef MakeList<tag::mean, tag::error>::type type;
            };

            template<typename base_type> 
            class AccumulatorImplementation<tag::histogram, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
                typedef typename histogram_type<value_type_loc>::type histogram_t;
                typedef typename mean_type<value_type_loc>::type mean_type;
                typedef AccumulatorImplementation<tag::histogram, base_type> ThisType;
                
                public:
                    AccumulatorImplementation<tag::histogram, base_type>(ThisType const & arg): base_type(arg)
                    
                    {}
                    template<typename ArgumentPack>
                    AccumulatorImplementation<tag::histogram, base_type>(ArgumentPack const & args
                                                 , typename boost::disable_if<
                                                                              boost::is_base_of<ThisType, ArgumentPack>
                                                                            , int
                                                                            >::type = 0
                                             ): base_type(args)
                    {}
                    
                    inline histogram_t const histogram() const 
                    {
                        //TODO: implement
                        return 272.15;
                    }
                    
                    inline void operator()(value_type_loc val) 
                    {
                        base_type::operator()(val);
                    }
                    inline ThisType& operator <<(value_type_loc val) 
                    {
                        (*this)(val);
                        return (*this);
                    }
                    
                    template<typename Stream>  inline void print(Stream & os) {
                        base_type::print(os);
                        os << "tag::histogram: " << std::endl;
                    }

                    inline void reset() {
                        base_type::reset();
                    }
                private:
            };

            template<typename base_type> class ResultImplementation<tag::histogram, base_type> : public base_type  {

                public:

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : base_type(accum)
                    {}
// TODO: implement!
            };


        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(histogram, tag::histogram)

    }
}
#endif //ALPS_NGS_ALEA_DETAIL_HISTOGRAM_IMPLEMENTATION_HEADER
