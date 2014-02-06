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

#ifndef ALPS_NGS_ALEA_DETAIL_FIX_SIZE_BINNING_HPP
#define ALPS_NGS_ALEA_DETAIL_FIX_SIZE_BINNING_HPP

#include <alps/ngs/alea/feature/mean.hpp>
#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>

#include <alps/ngs/alea/accumulator/arguments.hpp>

#include <vector>
#include <ostream>
#include <cmath>
#include <algorithm>

namespace alps
{
    namespace accumulator
    {
        //=================== fixed_size_binning proxy ===================
        template<typename value_type>
        class fixed_size_binning_proxy_type
        {
            typedef typename mean_type<value_type>::type mean_type;
            typedef typename std::vector<value_type>::size_type size_type;
        public:
            fixed_size_binning_proxy_type(): bin_(std::vector<mean_type>()) {}
            fixed_size_binning_proxy_type(  std::vector<mean_type> const & bin
                                      , size_type const & bin_size):
                                                                  bin_(bin)
                                                                , bin_size_(bin_size)
            {}
            
            inline std::vector<mean_type> const & bins() const 
            {
                return bin_;
            }
            
            inline size_type const & bin_size() const
            {
                return bin_size_;
            }
            
            template<typename T>
            friend std::ostream & operator<<(std::ostream & os, fixed_size_binning_proxy_type<T> const & arg);
        private:
            std::vector<mean_type> const & bin_;
            size_type bin_size_;
        };

        template<typename T>
        inline std::ostream & operator<<(std::ostream & os, fixed_size_binning_proxy_type<T> const & arg)
        {
            os << "fixed_size_binning_proxy" << std::endl;
            return os;
            
        };
        //=================== fixed_size_binning trait ===================
        template <typename T> struct fixed_size_binning_type {
            typedef fixed_size_binning_proxy_type<T> type;
        };
        //=================== fixed_size_binning implementation ===================
        namespace detail {
            //set up the dependencies for the tag::fixed_size_binning-Implementation
            template<> 
            struct Dependencies<tag::fixed_size_binning> 
            {
                typedef MakeList<tag::mean, tag::error>::type type;
            };

            template<typename base_type> 
            class AccumulatorImplementation<tag::fixed_size_binning, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
                typedef typename fixed_size_binning_type<value_type_loc>::type fix_bin_type;
                typedef typename std::vector<value_type_loc>::size_type size_type;
                typedef typename mean_type<value_type_loc>::type mean_type;
                typedef AccumulatorImplementation<tag::fixed_size_binning, base_type> ThisType;
                    
                public:
                    AccumulatorImplementation<tag::fixed_size_binning, base_type>(ThisType const & arg):  base_type(arg)
                                                                        , bin_(arg.bin_)
                                                                        , partial_(arg.partial_)
                                                                        , partial_count_(arg.partial_count_)
                                                                        , bin_size_(arg.bin_size_) 
                    {}
                    
                    // TODO: set right default value
                    template<typename ArgumentPack>
                    AccumulatorImplementation<tag::fixed_size_binning, base_type>(ArgumentPack const & args
                                             , typename boost::disable_if<
                                                                          boost::is_base_of<ThisType, ArgumentPack>
                                                                        , int
                                                                         >::type = 0
                                            ): base_type(args)
                                             , partial_()
                                             , partial_count_(0)
                                             , bin_size_(args[bin_size | 128]) //change doc if modified
                    {}
                    
                    inline fix_bin_type const fixed_size_binning() const 
                    { 
                        return fixed_size_binning_proxy_type<value_type_loc>(bin_, bin_size_); 
                    }
              
                    inline void operator()(value_type_loc const & val) 
                    {
                        using namespace alps::ngs::numeric;
                        using alps::ngs::numeric::detail::check_size;
                        
                        base_type::operator()(val);
                        
                        check_size(partial_, val);
                        partial_ += val;
                        ++partial_count_;
                        
                        if(partial_count_ == bin_size_)
                        {
                            bin_.push_back(partial_ / (typename alps::hdf5::scalar_type<value_type_loc>::type)bin_size_);
                            partial_count_ = 0;
                            partial_ = value_type_loc();
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
                        os << "FixBinSize: BinNumber: " << bin_.size() << " " << std::endl;
                        
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
                        partial_ = value_type_loc();
                        partial_count_ = 0;
                    }
                private:
                    std::vector<mean_type> bin_;
                    value_type_loc partial_;
                    size_type partial_count_;
                    size_type const bin_size_;
            };

            template<typename base_type> class ResultImplementation<tag::fixed_size_binning, base_type> : public base_type  {

                public:

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : base_type(accum)
                    {}
// TODO: implement!
            };


        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(fixed_size_binning, tag::fixed_size_binning)

    }
}
#endif
