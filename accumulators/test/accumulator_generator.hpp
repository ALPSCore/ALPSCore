/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file accumulator_generator.hpp
    Generators for accumulators and results of given types.
*/

#ifndef ALPS_ACCUMULATORS_TESTS_ACCUMULATOR_GENERATOR_HPP_INCLUDED
#define ALPS_ACCUMULATORS_TESTS_ACCUMULATOR_GENERATOR_HPP_INCLUDED

#include <cmath>

namespace alps {
    namespace accumulators {
        namespace testing {

            // Functions to compare vector or scalar values
            /// Compare vector values
            template <typename T>
            inline void compare_near(const std::vector<T>& expected, const std::vector<T>& actual, double tol, const std::string& descr)
            {
                EXPECT_EQ(expected.size(), actual.size()) << "Sizes of "+descr+" differ";
                for (int i=0; i<expected.size(); ++i) {
                    EXPECT_NEAR(expected.at(i), actual.at(i), tol) << "Element #"+boost::lexical_cast<std::string>(i)+" of "+descr+" differs";
                }
            }

            /// Compare scalar values
            template <typename T>
            inline void compare_near(T expected, T actual, double tol, const std::string& descr)
            {
                EXPECT_NEAR(expected, actual, tol) << "Values of "+descr+" differ";
            }

            /// Get scalar data point 
            template <typename T>
            inline T get_data(T val) { return val; }

            /// Get vector data point
            template <typename T>
            inline T get_data(typename T::value_type val)
            {
                return T(10, val);
            }


            /// Class to generate accumulators and results of a given type
            template <typename A, unsigned long int NPOINTS_PARAM=10000>
            class AccResultGenerator  {
                private:
                alps::accumulators::result_set* results_ptr_;
                alps::accumulators::accumulator_set* measurements_ptr_;
                const std::string name_;
    
                public:
                typedef A named_acc_type;
                typedef typename alps::accumulators::value_type<typename named_acc_type::accumulator_type>::type value_type;

                static const unsigned long int NPOINTS=NPOINTS_PARAM; /// < Number of data points
                static double tol() { return 5.E-3; }         /// < Recommended tolerance to compare expected and actual results (FIXME: should depend on NPOINTS)
                /// Free the memory allocated in the constructor
                virtual ~AccResultGenerator()
                {
                    delete results_ptr_;
                    delete measurements_ptr_;
                }
    
                /// Generate the data points for the accumulator
                AccResultGenerator() : name_("acc")
                {
                    srand48(43);
                    measurements_ptr_=new alps::accumulators::accumulator_set();
                    alps::accumulators::accumulator_set& m=*measurements_ptr_;
                    m << named_acc_type(name_);
                    for (int i=0; i<NPOINTS; ++i) {
                        double d=drand48();
                        m[name_] << get_data<value_type>(d);
                    }
                    results_ptr_=new alps::accumulators::result_set(m);
                }

                /// Returns extracted results
                const alps::accumulators::result_wrapper& result() const
                {
                    return (*results_ptr_)[name_];
                }

                /// Returns the underlying accumulator
                const alps::accumulators::accumulator_wrapper& accumulator() const
                {
                    return (*measurements_ptr_)[name_];
                }

                /// Returns the expected mean
                double expected_mean() const { return 0.5; }
                /// Returns the expected error
                double expected_err() const { return 1/(12*std::sqrt(NPOINTS-1.)); }
            };

        } // tesing::
    } // accumulators::
} // alps::


#endif /* ALPS_ACCUMULATORS_TESTS_ACCUMULATOR_GENERATOR_HPP_INCLUDED */
