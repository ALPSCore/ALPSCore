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


#ifndef ALPS_NGS_ALEA_DETAIL_FIX_SIZE_BIN_ADAPTER_HEADER
#define ALPS_NGS_ALEA_DETAIL_FIX_SIZE_BIN_ADAPTER_HEADER

#include <alps/ngs/alea/accumulator_impl.hpp>

namespace alps
{
    namespace alea
    {
        namespace detail
        {
            //set up the dependencies for the FixSizeBinning-Adapter
            template<> 
            struct Dependencies<FixSizeBinning> 
            {
                typedef MakeList<Mean, Error>::type type;
            };

            template<typename base> 
            class Adapter<FixSizeBinning, base> : public base 
            {
                typedef typename fix_size_bin_type<typename base::value_type>::type fix_bin_type;
                typedef typename std::vector<typename base::value_type>::size_type size_type;
                typedef Adapter<FixSizeBinning, base> ThisType;
                    
                public:
                    Adapter<FixSizeBinning, base>(ThisType const & arg):  base(arg)
                                                                        , bin_(arg.bin_)
                                                                        , partial_(arg.partial_)
                                                                        , partial_count_(arg.partial_count_)
                                                                        , bin_size_(arg.bin_size_) 
                    {}
                    
                    // TODO: set right default value
                    template<typename ArgumentPack>
                    Adapter<FixSizeBinning, base>(ArgumentPack const & args
                                             , typename boost::disable_if<
                                                                          boost::is_base_of<ThisType, ArgumentPack>
                                                                        , int
                                                                         >::type = 0
                                            ): base(args)
                                             , partial_()
                                             , partial_count_(0)
                                             , bin_size_(args[bin_size | 128]) 
                    {}
                    
                    fix_bin_type fix_size_bin() const 
                    { 
                        //TODO: Implementation
                        return bin_size_; 
                    }
              
                    ThisType& operator <<(typename base::value_type val) 
                    {
                        base::operator << (val);
                        
                        partial_ += val;
                        ++partial_count_;
                        
                        if(partial_count_ == bin_size_)
                        {
                            bin_.push_back(partial_);
                            partial_count_ = 0;
                            partial_ = typename base::value_type();
                        }
                        return *this;
                    }
              
                    template<typename Stream> 
                    void print(Stream & os) 
                    {
                        base::print(os);
                        os << "FixBinSize: " << fix_size_bin() << " " << "BinNumber: " << bin_.size() << " ";
                    }
                private:
                    std::vector<typename base::value_type> bin_;
                    typename base::value_type partial_;
                    size_type partial_count_;
                    size_type bin_size_;
            };
        } // end namespace detail
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_FIX_SIZE_BIN_ADAPTER
