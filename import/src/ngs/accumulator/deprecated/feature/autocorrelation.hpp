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

#ifndef ALPS_NGS_ALEA_DETAIL_AUTOCORRELATION_IMPLEMENTATION_HEADER
#define ALPS_NGS_ALEA_DETAIL_AUTOCORRELATION_IMPLEMENTATION_HEADER

#include <alps/ngs/alea/feature/mean.hpp>
#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>

#include <alps/ngs/numeric/array.hpp>
#include <alps/ngs/numeric/detail.hpp>
#include <alps/ngs/numeric/vector.hpp>

#include <alps/multi_array.hpp>

#include <boost/cstdint.hpp>

#include <ostream>
#include <algorithm>
#include <vector>
#include <cmath>

namespace alps
{
    namespace accumulator
    {
        //=================== autocorrelation proxy ===================
        template<typename value_type>
        class autocorrelation_proxy_type
        {
            typedef typename mean_type<value_type>::type mean_type;
        public:
            autocorrelation_proxy_type(): bin2_(std::vector<value_type>())
                                , bin1_(std::vector<value_type>())
                                , count_(0) 
            {}
            autocorrelation_proxy_type(  std::vector<value_type> const & bin2
                                , std::vector<value_type> const & bin1
                                , boost::uint64_t const & count):
                                                                  bin2_(bin2)
                                                                , bin1_(bin1)
                                                                , count_(count)
            {}
            
            inline std::vector<value_type> const & bins() const 
            {
                return bin2_;
            }
            inline std::vector<value_type> const & sum() const 
            {
                return bin1_;
            }
            
            inline mean_type const error(boost::uint64_t level = -1) const
            {
                using namespace alps::ngs::numeric;
                //~ using alps::ngs::numeric::operator*;
                //~ using alps::ngs::numeric::operator-;
                //~ using alps::ngs::numeric::operator/;
                if(level == -1)
                    level = std::max(bin2_.size() - 5, typename std::vector<value_type>::size_type(0));
                using std::sqrt;
                using alps::ngs::numeric::sqrt;
                return sqrt((bin2_[level] - bin1_[level] * bin1_[level]) / (count_ - 1));
            }
            
            template<typename T>
            friend std::ostream & operator<<(std::ostream & os, autocorrelation_proxy_type<T> const & arg);
        private:
            std::vector<value_type> const & bin2_;
            std::vector<value_type> const & bin1_;
            boost::uint64_t const count_;
        };
        
        template<typename T>
        inline std::ostream & operator<<(std::ostream & os, autocorrelation_proxy_type<T> const & arg)
        {
            os << "autocorrelation_proxy" << std::endl;
            return os;
            
        };
        //=================== autocorrelation trait ===================
        template <typename T>
        struct autocorrelation_type
        {
            typedef autocorrelation_proxy_type<T> type;
        };
        //=================== autocorrelation implementation ===================
        namespace detail
        {

        //set up the dependencies for the tag::autocorrelationelation-Implementation
            template<> 
            struct Dependencies<tag::autocorrelation> 
            {
                typedef MakeList<tag::mean, tag::error, tag::detail::tau, tag::detail::converged>::type type;
            };

            template<typename base_type> 
            class AccumulatorImplementation<tag::autocorrelation, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
                typedef typename autocorrelation_type<value_type_loc>::type autocorrelation_type;
                typedef typename std::vector<value_type_loc>::size_type size_type;
                typedef typename mean_type<value_type_loc>::type mean_type;
                typedef AccumulatorImplementation<tag::autocorrelation, base_type> ThisType;
                
                public:
                    AccumulatorImplementation<tag::autocorrelation, base_type>(ThisType const & arg): base_type(arg)
                                                                        , bin_(arg.bin_)
                                                                        , partial_(arg.partial_)
                                                                        , bin_size_now_(arg.bin_size_now_)
                    {}
                    
                    template<typename ArgumentPack>
                    AccumulatorImplementation<tag::autocorrelation, base_type>(ArgumentPack const & args
                                                 , typename boost::disable_if<
                                                                              boost::is_base_of<ThisType, ArgumentPack>
                                                                            , int
                                                                            >::type = 0
                                             ): base_type(args)
                                              , bin_size_now_(0)
                    {}
                    
                    inline autocorrelation_type const autocorrelation() const 
                    { 
                        return autocorrelation_proxy_type<value_type_loc>(bin_, partial_, base_type::count());
                    }
                    
                    inline void operator()(value_type_loc const & val) 
                    {
                        using namespace alps::ngs::numeric;
                        using alps::ngs::numeric::operator+;
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator-;
                        
                        base_type::operator()(val);
                        
                        
                        if(base_type::count() == (1 << bin_size_now_))
                        {
                            bin_.push_back(value_type_loc());
                            partial_.push_back(value_type_loc());
                            ++bin_size_now_;
                        }
                        for (unsigned i = 0; i < bin_size_now_; ++i)
                        {
                            if(base_type::count() % (1lu<<i) == 0)
                            {
                                partial_[i] = base_type::sum_ - partial_[i];
                                bin_[i] += partial_[i]*partial_[i];
                                partial_[i] = base_type::sum_;
                            }
                        }
                    }
                    inline ThisType& operator<<(value_type_loc const & val) 
                    {
                        (*this)(val);
                        return (*this);
                    }
                    
                    template<typename Stream> 
                    inline void print(Stream & os) 
                    {
                        base_type::print(os);
                        os << "tag::autocorrelation: " << std::endl;
                        
                        //~ os << std::endl;
                        //~ for (unsigned int i = 0; i < bin_.size(); ++i)
                        //~ {
                            //~ os << "bin[" << i << "] = " << bin_[i] << std::endl;
                        //~ }
                    }
                    
                    inline void reset()
                    {
                        base_type::reset();
                        bin_.clear();
                        partial_.clear();
                        bin_size_now_ = 0;
                    }
                    
                private:
                    std::vector<value_type_loc> bin_;
                    std::vector<value_type_loc> partial_;
                    size_type bin_size_now_;
            };

            template<typename base_type> class ResultImplementation<tag::autocorrelation, base_type> : public base_type  {

                public:

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : base_type(accum)
                    {}
// TODO: implement!
            };
        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(autocorrelation, tag::autocorrelation)

    }
}
#endif
