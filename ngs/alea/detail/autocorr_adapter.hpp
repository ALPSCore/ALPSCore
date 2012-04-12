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


#ifndef ALPS_NGS_ALEA_DETAIL_AUTOCORR_ADAPTER_HEADER
#define ALPS_NGS_ALEA_DETAIL_AUTOCORR_ADAPTER_HEADER

#include <alps/ngs/alea/accumulator_impl.hpp>

#include <boost/cstdint.hpp>
#include <boost/lambda/lambda.hpp>

#include <vector>
#include <cmath>

namespace alps
{
    namespace alea
    {
        namespace detail
        {

        //set up the dependencies for the Autocorrelation-Adapter
            template<> 
            struct Dependencies<Autocorrelation> 
            {
                typedef MakeList<Mean, Error>::type type;
            };

            template<typename base> 
            class Adapter<Autocorrelation, base> : public base 
            {
                typedef typename autocorr_type<typename base::value_type>::type auto_bin_type;
                typedef typename std::vector<typename base::value_type>::size_type size_type;
                typedef Adapter<Autocorrelation, base> ThisType;
                
                public:
                    Adapter<Autocorrelation, base>(ThisType const & arg): base(arg)
                                                                        , bin_(arg.bin_)
                                                                        , partial_(arg.partial_)
                                                                        , bin_size_now_(arg.bin_size_now_)
                                                                        , pos_now_(arg.pos_now_)
                    {}
                    
                    template<typename ArgumentPack>
                    Adapter<Autocorrelation, base>(ArgumentPack const & args
                                                 , typename boost::disable_if<
                                                                              boost::is_base_of<ThisType, ArgumentPack>
                                                                            , int
                                                                            >::type = 0
                                             ): base(args)
                                              , partial_(1, typename base::value_type())
                                              , bin_size_now_(0)
                                              , pos_now_(0)
                    {}
                    
                    auto_bin_type autocorr() const 
                    { 
                        //TODO: implementation
                        return 0;
                    }
                    
                    ThisType& operator <<(typename base::value_type val) 
                    {
                        base::operator<<(val);
                        
                        //TODO: Right implementation
                        if(base::count() == (1 << bin_size_now_))
                        {
                            bin_.push_back(typename base::value_type());
                            partial_.push_back(partial_.back());
                            ++bin_size_now_;
                            pos_now_ = 0;
                            
                        }
                        ++pos_now_;
                        for (unsigned i = 0; i < bin_size_now_; ++i)
                        {
                            partial_[i] += val;
                            
                            if(pos_now_ % (1<<i) == 0)
                            {
                              bin_[i] += partial_[i]*partial_[i];
                                partial_[i] = 0;
                            }
                        }
                        
                        return *this;
                    }
                    
                    template<typename Stream> 
                    void print(Stream & os) 
                    {
                        base::print(os);
                        os << "Autocorrelation: ";
                        os << autocorr() << "\t";
                        
                        for_each(bin_.begin(), bin_.end(), (os << boost::lambda::_1 << " "));
                    }
                    
                private:
                    std::vector<typename base::value_type> bin_;
                    std::vector<typename base::value_type> partial_;
                    size_type bin_size_now_;
                    size_type pos_now_;
            };
        } // end namespace detail
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_AUTOCORR_ADAPTER
