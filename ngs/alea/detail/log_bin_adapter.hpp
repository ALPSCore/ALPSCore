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


#ifndef ALPS_NGS_ALEA_DETAIL_LOG_BIN_ADAPTER_HEADER
#define ALPS_NGS_ALEA_DETAIL_LOG_BIN_ADAPTER_HEADER

#include <alps/ngs/alea/accumulator_impl.hpp>

#include <boost/cstdint.hpp>

#include <vector>
namespace alps
{
    namespace alea
    {
        namespace detail
        {
            //set up the dependencies for the LogBinning-Adapter
            template<> 
            struct Dependencies<LogBinning> 
            {
                typedef MakeList<Mean, Error>::type type;
            };

            template<typename base> 
            class Adapter<LogBinning, base> : public base 
            {
                typedef typename log_bin_type<typename base::value_type>::type log_bin_type;
                typedef typename std::vector<typename base::value_type>::size_type size_type;
                typedef Adapter<LogBinning, base> ThisType;
          
                public:    
                    Adapter<LogBinning, base>(ThisType const & arg):  base(arg)
                                                                    , bin_(arg.bin_)
                                                                    , partial_(arg.partial_)
                                                                    , pos_in_partial_(arg.pos_in_partial_)
                                                                    , bin_size_now_(arg.bin_size_now_) 
                    {}
                    
                    //TODO: check if parameter is needed and if so, set right default value
                    template<typename ArgumentPack>
                    Adapter<LogBinning, base>(ArgumentPack const & args
                                         , typename boost::disable_if<
                                                                      boost::is_base_of<ThisType, ArgumentPack>
                                                                    , int
                                                                    >::type = 0
                                        ): base(args)
                                         , partial_()
                                         , pos_in_partial_()
                                         , bin_size_now_(1)
                    {}
                    
                    log_bin_type log_bin() const 
                    { 
                        //TODO: Implementation
                        return bin_.size(); 
                    }
              
                    ThisType& operator <<(typename base::value_type val) 
                    {
                        base::operator <<(val);
                        
                        //TODO: Implementation
                        partial_ += val;
                        ++pos_in_partial_;
                        
                        if(pos_in_partial_ == bin_size_now_)
                        {
                            bin_.push_back(partial_);
                            partial_ = typename base::value_type();
                            pos_in_partial_ = 0;
                            bin_size_now_ *= 2;
                        }
                        return *this;
                    }
              
                    template<typename Stream>
                    void print(Stream & os) 
                    {
                        
                        base::print(os);
                        os << "Log Binning: ";
                        os << log_bin();
                        //~ using namespace boost::lambda;
                        //~ for_each(bin_.begin(), bin_.end(), (os << _1 << " "));
                    }
              
                private:
                    std::vector<typename base::value_type> bin_;
                    typename base::value_type partial_;
                    size_type pos_in_partial_;
                    size_type bin_size_now_;
            };
        } // end namespace detail
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_LOG_BIN_ADAPTER
