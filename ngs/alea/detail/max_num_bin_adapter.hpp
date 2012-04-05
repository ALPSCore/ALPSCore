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


#ifndef ALPS_NGS_ALEA_DETAIL_MAX_NUM_BIN_ADAPTER_HEADER
#define ALPS_NGS_ALEA_DETAIL_MAX_NUM_BIN_ADAPTER_HEADER

#include <alps/ngs/alea/accumulator_impl.hpp>

#include <vector>
namespace alps
{
    namespace alea
    {
        namespace detail
        {
            //set up the dependencies for the MaxNumberBinning-Adapter
            template<>
            struct Dependencies<MaxNumberBinning> 
            {
                typedef MakeList<Mean, Error>::type type;
            };

            template<typename base> 
            class Adapter<MaxNumberBinning, base> : public base 
            {
                typedef typename max_num_bin_type<typename base::value_type>::type num_bin_type;
                typedef typename std::vector<typename base::value_type>::size_type size_type;
                typedef Adapter<MaxNumberBinning, base> ThisType;

                public:
                    Adapter<MaxNumberBinning, base>(ThisType const & arg): base(arg)
                                                                  , bin_(arg.bin_)
                                                                  , partial_(arg.partial_)
                                                                  , elements_in_bin_(arg.elements_in_bin_)
                                                                  , pos_in_partial_(arg.pos_in_partial_)
                                                                  , max_bin_number_(arg.max_bin_number_) 
                    {}
                    //TODO: set right default value 
                    
                    template<typename ArgumentPack>
                    Adapter<MaxNumberBinning, base>(ArgumentPack const & args
                                                , typename boost::disable_if<
                                                                              boost::is_base_of<ThisType, ArgumentPack>
                                                                            , int
                                                                            >::type = 0
                                               ): base(args)
                                                , partial_()
                                                , elements_in_bin_(1)
                                                , pos_in_partial_(0)
                                                , max_bin_number_(args[bin_number | 128])
                    {}
                    
                    num_bin_type max_num_bin() const { return max_bin_number_;};
              
                    ThisType& operator <<(typename base::value_type val)
                    {
                        base::operator <<(val);
                        
                        partial_ = partial_ + val;
                        ++pos_in_partial_;
                        
                        if(pos_in_partial_ == elements_in_bin_)
                        {
                            if(bin_.size() >= max_bin_number_)
                            {
                                if(max_bin_number_ % 2 == 1)
                                {
                                    partial_ += bin_[max_bin_number_ - 1];
                                    pos_in_partial_ += elements_in_bin_;
                                }
                                
                                for(unsigned int i = 0; i < max_bin_number_ / 2; ++i) //the rounding down here is intentional
                                    bin_[i] = bin_[2*i] + bin_[2*i + 1];
                                
                                bin_.erase(bin_.begin() + max_bin_number_ / 2, bin_.end());
                                
                                elements_in_bin_ *= 2;
                            }
                            if(pos_in_partial_ == elements_in_bin_)
                            {
                                bin_.push_back(partial_);
                                partial_ = typename base::value_type();
                                pos_in_partial_ = 0;
                            }
                        }
                        return *this;
                    }
              
                    template<typename Stream> 
                    void print(Stream & os) 
                    {
                        base::print(os);
                        os << "MaxBinningNumber: " << max_num_bin() << std::endl;
                        
                        for (unsigned int i = 0; i < bin_.size(); ++i)
                        {
                            os << "bin[" << i << "] = " << bin_[i] << std::endl;
                        }
                    }
              
                private:
                    std::vector<typename base::value_type> bin_;
                    typename base::value_type partial_;
                    size_type elements_in_bin_;
                    size_type pos_in_partial_;
                    size_type max_bin_number_;
            };
        } // end namespace detail
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_MAX_NUM_BIN_ADAPTER
