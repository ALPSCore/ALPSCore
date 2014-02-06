/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                              Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_ALEA_WEIGHT_HPP
#define ALPS_NGS_ALEA_WEIGHT_HPP

#include <alps/ngs/alea/feature/mean.hpp>
#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>
#include <alps/ngs/alea/feature/tags.hpp>

#include <alps/ngs/alea/accumulator/arguments.hpp>

#include <boost/shared_ptr.hpp>

#include <vector>
#include <ostream>
#include <cmath>
#include <algorithm>

namespace alps {
    namespace accumulator {
        template<
              typename vt
            , typename features_input
            , typename wvt
        >
        class accumulator;
        
        namespace detail {
            class accumulator_wrapper;
        }
        //=================== weight proxy ===================
        //=================== weight trait ===================
        template <typename T> struct weight_type {
            typedef double type;
        };
        //=================== weight implementation ===================
        namespace detail {
            //set up the dependencies for the tag::detail::weight-Implementation
            template<> 
            struct Dependencies<tag::detail::weight> 
            {
                typedef MakeList<tag::mean, tag::error>::type type;
            };

            template<typename base_type> 
            class AccumulatorImplementation<tag::detail::weight, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
                typedef typename base_type::weight_value_type weight_value_type_loc;
                typedef typename weight_type<value_type_loc>::type weight_type_loc; //don't mix up weight_value and weight type!!
                typedef typename std::vector<value_type_loc>::size_type size_type;
                typedef AccumulatorImplementation<tag::detail::weight, base_type> ThisType;
                    
                public:
                    typedef ThisType weight_tag_type; //technical use only / see impl_picker in derived_wrapper
                    AccumulatorImplementation<tag::detail::weight, base_type>(ThisType const & arg):  base_type(arg)
                                                                        , ownes_weight_acc_(arg.ownes_weight_acc_)
                                                                        , weight_acc_ptr_(arg.weight_acc_ptr_)
                    {}
                    
                    template<typename ArgumentPack>
                    AccumulatorImplementation<tag::detail::weight, base_type>(ArgumentPack const & args
                                             , typename boost::disable_if<
                                                                          boost::is_base_of<ThisType, ArgumentPack>
                                                                        , int
                                                                         >::type = 0
                                            ): base_type(args)
                                             , ownes_weight_acc_(false)
                                             //~ , weight_acc_ptr_(NULL)
                                             //~ , weight_acc_ptr_(args[weight_ref| boost::shared_ptr<accumulator_wrapper>()])
                                                                                          
                    {
                        if(weight_acc_ptr_ == NULL)
                        {
                            ownes_weight_acc_ = true;
                            //TODO: features?
                            //~ weight_acc_ptr_ = new accumulator_wrapper(accumulator<weight_type_loc, features<tag::mean, tag::error>, void>());
                            
                        }
                    }
                    
                    inline weight_type_loc const weight() const {
                        return weight_type_loc(); 
                    }
                    
                    inline void operator()(value_type_loc const & val, weight_value_type_loc const & w) {
                        using namespace alps::ngs::numeric;
                        
                        base_type::operator()(val * w);
                    }

                    template<typename ArgumentPack>
                    inline void operator()(value_type_loc const & val, ArgumentPack & arg) {
                        using namespace alps::ngs::numeric;
                        base_type::operator()(val * arg[Weight]);
                    }
                    
                    inline void operator()(value_type_loc const & val) {
                        using namespace alps::ngs::numeric;
                        base_type::operator()(val * weight_type_loc(1));
                    }

                    inline ThisType& operator<<(value_type_loc const & val) {
                        (*this)(val);
                        return (*this);
                    }

                    template<typename Stream> inline void print(Stream & os)  {
                        base_type::print(os);
                        os << "Weight: " << ownes_weight_acc_ << std::endl;
                    }

                    void save(hdf5::archive & ar) const {
                        base_type::save(ar);
                        ar["ownsweight"] = ownes_weight_acc_;
                        // TODO: how do we handle shared weights?
                    }

                    void load(hdf5::archive & ar) {
                        using alps::ngs::numeric::operator*;

                        base_type::load(ar);
                        ar["ownsweight"] >> ownes_weight_acc_;
                        // TODO: how do we handle shared weights?
                    }

                    inline void reset() {
                    
                    }

                private:

                    bool ownes_weight_acc_;
                    boost::shared_ptr<accumulator_wrapper> weight_acc_ptr_;
            };

            template<typename base_type> class ResultImplementation<tag::detail::weight, base_type> : public base_type  {

                public:

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : base_type(accum)
                    {}
// TODO: implement!
            };

        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(weight, tag::detail::weight)

    }
}
#endif
