/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_DETAIL_TAU_IMPLEMENTATION_HEADER
#define ALPS_NGS_ALEA_DETAIL_TAU_IMPLEMENTATION_HEADER

#include <alps/ngs/alea/feature/mean.hpp>
#include <alps/ngs/alea/feature/feature_traits.hpp>
#include <alps/ngs/alea/feature/generate_property.hpp>

#include <boost/cstdint.hpp>

#include <vector>
#include <cmath>

namespace alps
{
    namespace accumulator
    {
        //=================== tau proxy ===================
        //=================== tau trait ===================
        template <typename T>
        struct tau_type
        {
            typedef double type;
        };
        //=================== tau implementation ===================
        namespace detail
        {
        //set up the dependencies for the tag::autocorrelation-Implementation
            template<> 
            struct Dependencies<tag::detail::tau> 
            {
                typedef MakeList<>::type type;
            };

            template<typename base_type> 
            class AccumulatorImplementation<tag::detail::tau, base_type> : public base_type 
            {
                typedef typename base_type::value_type value_type_loc;
                typedef typename tau_type<value_type_loc>::type tau_type;
                typedef typename mean_type<value_type_loc>::type mean_type;
                typedef AccumulatorImplementation<tag::detail::tau, base_type> ThisType;
                
                public:
                    AccumulatorImplementation<tag::detail::tau, base_type>(ThisType const & arg): base_type(arg)
                    {}
                    
                    template<typename ArgumentPack>
                    AccumulatorImplementation<tag::detail::tau, base_type>(ArgumentPack const & args
                                                 , typename boost::disable_if<
                                                                              boost::is_base_of<ThisType, ArgumentPack>
                                                                            , int
                                                                            >::type = 0
                                             ): base_type(args)
                    {}
                    
                    inline tau_type const tau() const 
                    {
                        //~ //Simplebinning.h Zeile 475
                        //~ template <class T>
                        //~ inline typename SimpleBinning<T>::time_type SimpleBinning<T>::tau() const
                        //~ {
                          //~ if (count()==0)
                            //~ boost::throw_exception(NoMeasurementstag::error());
                        //~ 
                          //~ if( binning_depth() >= 2 )
                          //~ {
                            //~ count_type factor =count()-1;
                            //~ time_type er(std::abs(error()));
                            //~ er *=er*factor;
                            //~ er /= std::abs(variance());
                            //~ er -=1.;
                            //~ return 0.5*er;
                          //~ }
                          //~ else
                          //~ {
                            //~ time_type retval;
                            //~ resize_same_as(retval,sum_[0]);
                            //~ retval=inf();
                            //~ return retval;
                          //~ }
                        //~ }

                        //TODO: implement
                        return 42;
                    }
                    
                    inline void operator()(value_type_loc const & val) 
                    {
                        base_type::operator()(val);
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
                        os << "tag::detail::tau: " << std::endl;
                    }
                    inline void reset()
                    {
                        base_type::reset();
                    }
                private:
            };

            template<typename base_type> class ResultImplementation<tag::detail::tau, base_type> : public base_type  {

                public:

                    template<typename Accumulator> ResultImplementation(Accumulator const & accum)
                        : base_type(accum)
                    {}
// TODO: implement!
            };

        }

        //=================== call GENERATE_PROPERTY macro ===================
        GENERATE_PROPERTY(tau, tag::detail::tau)

    }
}
#endif //ALPS_NGS_ALEA_DETAIL_TAU_IMPLEMENTATION_HEADER
