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


#ifndef ALPS_NGS_ALEA_DETAIL_ERROR_ADAPTER_HEADER
#define ALPS_NGS_ALEA_DETAIL_ERROR_ADAPTER_HEADER

#include <alps/ngs/alea/accumulator_impl.hpp>

#include <cmath>
namespace alps
{
    namespace alea
    {
        namespace detail
        {
            //set up the dependencies for the Error-Adapter
            template<> 
            struct Dependencies<Error> 
            {
                typedef MakeList<Mean>::type type;
            };

            template<typename base> 
            class Adapter<Error, base> : public base 
            {
                typedef typename error_type<typename base::value_type>::type error_type;
                typedef Adapter<Error, base> ThisType;
                
                public:
                    Adapter<Error, base>(ThisType const & arg): base(arg), mean2_(arg.mean2_) {}
                    
                    template<typename ArgumentPack>
                    Adapter<Error, base>(ArgumentPack const & args, typename boost::disable_if<
                                                                                              boost::is_base_of<ThisType, ArgumentPack>
                                                                                            , int
                                                                                            >::type = 0
                                        ): base(args)
                                         , mean2_() 
                    {}
                    
                    error_type error() const 
                    { 
                        using std::sqrt;
                        return sqrt((mean2_ - base::mean()*base::mean()) / ((base::count() - 1)));
                    } 
                    
                    ThisType& operator <<(typename base::value_type val) 
                    {
                        base::operator <<(val);
                        mean2_ = ((base::count()-1) * mean2_ + val*val)/base::count();
                        return *this;
                    }
                    
                    template<typename Stream> 
                    void print(Stream & os) 
                    {
                        base::print(os);
                        os << "Error: " << error() << " ";
                    }
                    
                private:
                    error_type mean2_;
            };
        } // end namespace detail
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_ERROR_ADAPTER
