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

#include <alps/ngs/alea/accumulator_impl.hpp>
namespace alps
{
    namespace alea
    {
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
                        return mean_type(sum_)/base_type::count();
                    }
            
                    inline ThisType& operator <<(value_type_loc val) 
                    {
                        base_type::operator <<(val);
                        sum_ += val;
                        return *this;
                    }
            
                    template<typename Stream> 
                    inline void print(Stream & os) 
                    {
                        base_type::print(os);
                        os << "tag::mean: " << mean() << " " << std::endl;
                    }
            
                //~ private:
                protected:
                    mean_type sum_;
            };
        } // end namespace detail
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_MEAN_IMPLEMENTATION
