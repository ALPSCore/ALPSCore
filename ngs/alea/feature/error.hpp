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


#ifndef ALPS_NGS_ALEA_DETAIL_ERROR_IMPLEMENTATION_HEADER
#define ALPS_NGS_ALEA_DETAIL_ERROR_IMPLEMENTATION_HEADER

#include <alps/ngs/alea/accumulator/accumulator_impl.hpp>
#include <alps/ngs/short_print.hpp>
#include <alps/ngs/alea/features.hpp>
#include <alps/ngs/numeric/vector.hpp>
#include <alps/ngs/numeric/array.hpp>

#include <cmath>
namespace alps
{
    namespace alea
    {
        //=================== error proxy ===================
        //=================== error trait ===================
        template <typename T>
        struct error_type
        {
            typedef typename mean_type<T>::type type;
        };
        //=================== error implementation ===================
        namespace detail
        {
            //set up the dependencies for the tag::error-Implementation
            template<> 
            struct Dependencies<tag::error> 
            {
                typedef MakeList<tag::mean>::type type;
            };

            template<typename base_type> 
            class Implementation<tag::error, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
                typedef typename error_type<value_type_loc>::type error_type;
                typedef Implementation<tag::error, base_type> ThisType;
                
                public:
                    Implementation<tag::error, base_type>(ThisType const & arg): base_type(arg), sum2_(arg.sum2_) {}
                    
                    template<typename ArgumentPack>
                    Implementation<tag::error, base_type>(ArgumentPack const & args, typename boost::disable_if<
                                                                                              boost::is_base_of<ThisType, ArgumentPack>
                                                                                            , int
                                                                                            >::type = 0
                                        ): base_type(args)
                                         , sum2_() 
                    {}
                    
                    inline error_type const error() const 
                    { 
                        using alps::ngs::numeric::sqrt;
                        using std::sqrt;
                        using alps::ngs::numeric::operator/;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator*;

                        return sqrt((sum2_ / base_type::count() - base_type::mean()*base_type::mean()) / ((base_type::count() - 1)));
                    }
                    
                    inline ThisType& operator <<(value_type_loc val) 
                    {
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::operator*;
                        
                        base_type::operator <<(val);
                        sum2_ += val*val;
                        return *this;
                    }
                    
                    template<typename Stream> 
                    inline void print(Stream & os) 
                    {
                        base_type::print(os);
                        os << "tag::error: " << alps::short_print(error()) << " " << std::endl;
                    }
                    inline void reset()
                    {
                        sum2_ = error_type();
                        base_type::reset();
                    }
                private:
                    error_type sum2_;
            };
        } // end namespace detail
    }//end alea namespace 
}//end alps namespace
#endif // ALPS_NGS_ALEA_DETAIL_ERROR_IMPLEMENTATION
