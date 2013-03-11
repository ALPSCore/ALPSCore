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

#ifndef ALPS_NGS_ALEA_WEIGHTED_HPP
#define ALPS_NGS_ALEA_WEIGHTED_HPP

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
        //=================== weighted proxy ===================
        //=================== weighted trait ===================
        template <typename T> struct weighted_type {
            typedef double type;
        };
        //=================== weighted implementation ===================
        namespace detail {
            //set up the dependencies for the tag::weighted-Implementation
            template<> 
            struct Dependencies<tag::weighted> 
            {
                typedef MakeList<tag::mean, tag::error>::type type;
            };

            template<typename base_type> 
            class AccumulatorImplementation<tag::weighted, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
                typedef typename weighted_type<value_type_loc>::type weighted_type_loc;
                typedef typename std::vector<value_type_loc>::size_type size_type;
                typedef AccumulatorImplementation<tag::weighted, base_type> ThisType;
                    
                public:
                    AccumulatorImplementation<tag::weighted, base_type>(ThisType const & arg):  base_type(arg)
                                                                        //~ , 
                    {}
                    
                    template<typename ArgumentPack>
                    AccumulatorImplementation<tag::weighted, base_type>(ArgumentPack const & args
                                             , typename boost::disable_if<
                                                                          boost::is_base_of<ThisType, ArgumentPack>
                                                                        , int
                                                                         >::type = 0
                                            ): base_type(args)
                                             //~ , bin_size_(args[bin_size | 128]) //change doc if modified
                    {}
                    
                    inline weighted_type_loc const weighted() const 
                    { 
                        return weighted_type_loc(); 
                    }
              
                    inline ThisType& operator()(value_type_loc const & val, double const & w) 
                    {
                        using namespace alps::ngs::numeric;
                        
                        base_type::operator()(val * w);
                        
                        return *this;
                    }
                    
                    inline ThisType& operator()(value_type_loc const & val) 
                    {
                        using namespace alps::ngs::numeric;
                        
                        base_type::operator()(val);
                        
                        
                        return *this;
                    }
                    inline ThisType& operator<<(value_type_loc const & val) 
                    {
                        return (*this)(val);
                    }
                    
                    template<typename Stream> 
                    inline void print(Stream & os) 
                    {
                        base_type::print(os);
                        os << "Weighted:" << std::endl;
                    }
                    inline void reset()
                    {
                    
                    }
                private:
                    
            };

            template<typename base_type> class ResultImplementation<tag::weighted, base_type> {
// TODO: implement!
            };

        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(weighted, tag::weighted)

    }
}
#endif
