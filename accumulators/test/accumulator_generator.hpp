/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file accumulator_generator.hpp
    Generators for accumulators and results of given types.
*/

#ifndef ALPS_ACCUMULATORS_TESTS_ACCUMULATOR_GENERATOR_HPP_INCLUDED
#define ALPS_ACCUMULATORS_TESTS_ACCUMULATOR_GENERATOR_HPP_INCLUDED

#include <cmath>
#include <cstddef>
#include <numeric>
#include <boost/lexical_cast.hpp>
#include "alps/accumulators.hpp"

namespace alps {
    namespace accumulators {
        namespace testing {

            /// Meta-predicate to check if accumulator A is a named accumulator AA.
            /** Usage:
                  typedef is_same_accumulator<MeanAccumulator<double>,MeanAccumulator> is_true_type;
                  typedef is_same_accumulator<NoBinningAccumulator<double>,LogBinningAccumulator> is_false_type;
            */
            template <typename A, template<typename> class AA>
            struct is_same_accumulator:
                public std::is_same<A,
                                      AA<typename value_type<typename A::accumulator_type>::type>
                                     >
            {};

            // Functions to compare vector or scalar values
            /// Compare vector values
            template <typename T>
            inline void compare_near(const std::vector<T>& expected, const std::vector<T>& actual, double tol, const std::string& descr)
            {
                EXPECT_EQ(expected.size(), actual.size()) << "Sizes of "+descr+" differ";
                for (size_t i=0; i<expected.size(); ++i) {
                    EXPECT_NEAR(expected.at(i), actual.at(i), tol) << "Element #"+boost::lexical_cast<std::string>(i)+" of "+descr+" differs";
                }
            }

            /// Compare scalar values
            template <typename T>
            inline void compare_near(T expected, T actual, double tol, const std::string& descr)
            {
                EXPECT_NEAR(expected, actual, tol) << "Values of "+descr+" differ";
            }

            /// Functor class generating random data [0,1)
            struct RandomData {
                explicit RandomData(double ini=43) { srand48(ini); }
                double operator()() const { return drand48(); }

                /// Expected mean of first n members
                double mean(std::size_t /*n*/) const { return 0.5; }

                /// Expected error bar of first n members
                double error(std::size_t n) const { return 1./(12*std::sqrt(n-1.)); }
            };

            /// Functor class generating constant data (as initialized, 0.5 by default)
            struct ConstantData {
                double ini_;
                explicit ConstantData(double ini=0.5) :ini_(ini) { }
                double operator()() const { return ini_; }

                /// Expected mean of first n members
                double mean(std::size_t /*n*/) const { return ini_; }

                /// Expected error bar of first n members
                double error(std::size_t /*n*/) const { return 0; }
            };

            /// Functor class generating alternating data (0.5 +/- initialized; initialized=0.5 by default)
            struct AlternatingData {
                double ini_;
                mutable unsigned int step_;

                explicit AlternatingData(double ini=0.5) :ini_(ini), step_(0) { }
                double operator()() const {
                    if (step_==0) {
                        step_=1;
                        return 0.5-ini_;
                    } else {
                        step_=0;
                        return 0.5+ini_;
                    }
                }
                /// Expected mean of first n members
                double mean(std::size_t n) const { if (n%2==0) return 0.5; else return 0.5-ini_/n; }

                /// Expected error bar of first n members
                double error(std::size_t n) const {
                    if (n%2==0)
                        return ini_/std::sqrt(n-1.);
                    else
                        return ini_*std::sqrt(n+1.)/n;
                }
            };

            /// Functor class generating linearly changing data (ini, 2*ini, 3*ini...) default ini=1
            struct LinearData {
                double ini_;
                mutable double val_;

                explicit LinearData(double ini=1) :ini_(ini), val_(0) { }
                double operator()() const {
                    val_ += ini_;
                    return val_;
                }

                /// Expected mean of first n members
                double mean(std::size_t n) const { return ini_*(1.+n)/2.; }

                /// Expected error bar of first n members
                double error(std::size_t n) const { return ini_*std::sqrt((1.+n)/12.); }
            };

            /// Functor class generating correlated data with correlation length L, using generator G (likely, RandomData)
            template <std::size_t L, typename G=RandomData>
            class CorrelatedData {
              public:
                typedef G generator_type;
                static const std::size_t CORRL=L;
              private:
                // FIXME: is mutable appropriate? is mutable necessary?
                mutable generator_type gen_;
                /// Series "memory", and the to-be-returned sample at buf_[0]
                mutable double buf_[L+1]; // L would be enough, but we don't want problems when L==0

                void init_() {
                    std::fill_n(buf_,CORRL+1,0);
                }
              public:
                explicit CorrelatedData(double ini) : gen_(ini) { init_(); }
                explicit CorrelatedData() : gen_() { init_(); }

                double operator()() const {
                    buf_[0]=std::accumulate(buf_+1,buf_+CORRL+1,gen_())/(CORRL+1);
                    double ret=buf_[0];
                    for (std::size_t j=CORRL; j>=1; --j) buf_[j]=buf_[j-1];
                    return ret;
                }

                /// Expected mean of first n members
                double mean(std::size_t n) const { return gen_.mean(n); }

                /// Expected (uncorrelated!) error bar of first n members
                double error(std::size_t n) const { return gen_.error(n); }
            };


            /// Class generating data points of a general (presumably scalar) type T
            /** Usage: `T x=gen_data<T>(val,sz);` */
            // (a class is used to allow partial template specialization)
            template <typename T>
            struct gen_data {
                typedef T value_type;
                T val_;
                gen_data(T val, unsigned int =0) : val_(val) {}
                operator value_type() const { return val_; }
                value_type value() const { return val_; }
            };

            /// Class generating data points of vector<T>
            /** Usage: `std::vector<T> v=gen_data<T>(T(val),sz);` */
            template <typename T>
            struct gen_data< std::vector<T> > {
                typedef std::vector<T> value_type;
                T val_;
                unsigned int vsz_;
                /// Generate a vector of size 3 by default.
                gen_data(T val, unsigned int vsz =3) : val_(val), vsz_(vsz) {}
                operator value_type() const { return value(); }
                value_type value() const { return value_type(vsz_,val_); }
            };

            /// Class to generate accumulators and results of a given type
            template <typename A, std::size_t NPOINTS_PARAM=10000, typename NG=RandomData>
            class AccResultGenerator  {
              private:
                alps::accumulators::result_set* results_ptr_;
                alps::accumulators::accumulator_set* measurements_ptr_;
                const std::string name_;
                NG number_generator;

              public:
                typedef A named_acc_type;
                typedef typename alps::accumulators::value_type<typename named_acc_type::accumulator_type>::type value_type;

                static const std::size_t NPOINTS=NPOINTS_PARAM; /// < Number of data points
                static double tol() { return 5.E-3; }         /// < Recommended tolerance to compare expected and actual results (FIXME: should depend on NPOINTS and NG)

              private:
                /// Generate the data points for the accumulator
                void init_() {
                    measurements_ptr_=new alps::accumulators::accumulator_set();
                    alps::accumulators::accumulator_set& m=*measurements_ptr_;
                    m << named_acc_type(name_);
                    for (std::size_t i=0; i<NPOINTS; ++i) {
                        double d=number_generator();
                        m[name_] << gen_data<value_type>(d).value();
                    }
                    results_ptr_=new alps::accumulators::result_set(m);
                }

              public:
                /// Free the memory allocated in the constructor
                virtual ~AccResultGenerator()
                {
                    delete results_ptr_;
                    delete measurements_ptr_;
                }

                /// Construct the accumulator/result generator with default seed value for the number generator
                AccResultGenerator() : name_("data") { init_(); }

                /// Construct the accumulator/result generator with a specified seed value for the number generator
                AccResultGenerator(double ini) : name_("data"), number_generator(ini) { init_(); }

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

                /// Returns result set
                const alps::accumulators::result_set& results() const
                {
                    return *results_ptr_;
                }

                /// Returns the accumulator set
                const alps::accumulators::accumulator_set& accumulators() const
                {
                    return *measurements_ptr_;
                }

                /// Returns the accumulator/result name in the set
                std::string name() const { return name_; }

                /// Returns the expected mean
                double expected_mean() const { return number_generator.mean(NPOINTS); }
                /// Returns the expected error
                double expected_err() const { return number_generator.error(NPOINTS); }
            };

            /// Class to generate a pair of accumulators with identical data: A<T> and A<vector<T>>
            template <template <typename> class A, typename T, std::size_t NPOINTS_P=1000, unsigned int VSZ_P=3, typename NG=RandomData>
            class acc_vs_pair_gen {
              private:
                alps::accumulators::result_set* results_ptr_;
                alps::accumulators::accumulator_set* measurements_ptr_;
                NG number_generator;

                public:
                typedef T scalar_data_type;
                typedef std::vector<T> vector_data_type;
                typedef A<scalar_data_type> scalar_acc_type;
                typedef A<vector_data_type> vector_acc_type;

                static const std::size_t NPOINTS=NPOINTS_P; /// < Number of data points
                static const unsigned int VSIZE=VSZ_P; /// size of the vector

                const std::string scal_name;
                const std::string vec_name;

              private:
                /// Generate the data points for the accumulator
                void init_()
                {
                    measurements_ptr_=new alps::accumulators::accumulator_set();
                    alps::accumulators::accumulator_set& m=*measurements_ptr_;
                    m << vector_acc_type(vec_name)
                      << scalar_acc_type(scal_name);

                    for (std::size_t i=0; i<NPOINTS; ++i) {
                        double d=number_generator();
                        m[vec_name]  << gen_data<vector_data_type>(d, VSIZE).value();
                        m[scal_name] << gen_data<scalar_data_type>(d).value();
                    }
                    results_ptr_=new alps::accumulators::result_set(m);
                }

              public:
                /// Free the memory allocated in the constructor
                virtual ~acc_vs_pair_gen()
                {
                    delete results_ptr_;
                    delete measurements_ptr_;
                }

                /// Generate the data points for the accumulator, seed number generator with a default value
                acc_vs_pair_gen(const std::string& sname, const std::string& vname) : scal_name(sname), vec_name(vname)
                {
                    init_();
                }

                /// Generate the data points for the accumulator, seed number generator with a specified value
                acc_vs_pair_gen(double ini, const std::string& sname, const std::string& vname) : number_generator(ini),
                                                                                                  scal_name(sname),
                                                                                                  vec_name(vname)
                { init_(); }

                /// Returns extracted result set
                const alps::accumulators::result_set& results() const
                {
                    return *results_ptr_;
                }

                /// Returns the accumulator set
                const alps::accumulators::accumulator_set& accumulators() const
                {
                    return *measurements_ptr_;
                }
            };


            /// Class to generate accumulators with identical, correlated data: Mean,NoBinning,LogBinning,FullBinning
            template <typename T, std::size_t NPOINTS_P=1000, std::size_t CORRL_P=10, unsigned int VSZ_P=3, typename NG=RandomData>
            class acc_correlated_gen {
              private:
                alps::accumulators::result_set* results_ptr_;
                alps::accumulators::accumulator_set* measurements_ptr_;
                CorrelatedData<CORRL_P,NG> number_generator;

              public:
                typedef T data_type;
                typedef alps::accumulators::MeanAccumulator<T> mean_acc_type;
                typedef alps::accumulators::NoBinningAccumulator<T> nobin_acc_type;
                typedef alps::accumulators::LogBinningAccumulator<T> logbin_acc_type;
                typedef alps::accumulators::FullBinningAccumulator<T> fullbin_acc_type;

                static const std::size_t NPOINTS=NPOINTS_P; /// < Number of data points
                static const unsigned int VSIZE=VSZ_P; /// size of the vector
                static const std::size_t CORRL=CORRL_P; /// correlation length (samples)

                const std::string mean_name;
                const std::string nobin_name;
                const std::string logbin_name;
                const std::string fullbin_name;

              private:
                /// Generate the data points for the accumulator
                void init_()
                {
                    measurements_ptr_=new alps::accumulators::accumulator_set();
                    alps::accumulators::accumulator_set& m=*measurements_ptr_;
                    m <<    mean_acc_type(   mean_name)
                      <<   nobin_acc_type(  nobin_name)
                      <<  logbin_acc_type( logbin_name)
                      << fullbin_acc_type(fullbin_name);

                    for (std::size_t i=0; i<NPOINTS; ++i) {
                        data_type sample=gen_data<data_type>(number_generator(), VSIZE);
                        m[   mean_name]  << sample;
                        m[  nobin_name]  << sample;
                        m[ logbin_name]  << sample;
                        m[fullbin_name]  << sample;
                    }
                    results_ptr_=new alps::accumulators::result_set(m);
                }

              public:
                /// Free the memory allocated in the constructor
                virtual ~acc_correlated_gen()
                {
                    delete results_ptr_;
                    delete measurements_ptr_;
                }

                /// Generate the data points for the accumulator, seed number generator with a default value
                acc_correlated_gen(const std::string& mean="mean", const std::string& nobin="nobin",
                                   const std::string& logbin="logbin", const std::string& fullbin="fullbin")
                    : mean_name(mean),
                      nobin_name(nobin),
                      logbin_name(logbin),
                      fullbin_name(fullbin)
                { init_(); }

                /// Generate the data points for the accumulator, seed number generator with a specified value
                acc_correlated_gen(double ini,
                                   const std::string& mean="mean", const std::string& nobin="nobin",
                                   const std::string& logbin="logbin", const std::string& fullbin="fullbin")
                    : number_generator(ini),
                      mean_name(mean),
                      nobin_name(nobin),
                      logbin_name(logbin),
                      fullbin_name(fullbin)
                { init_(); }

                /// Returns extracted result set
                const alps::accumulators::result_set& results() const
                {
                    return *results_ptr_;
                }

                /// Returns the accumulator set
                const alps::accumulators::accumulator_set& accumulators() const
                {
                    return *measurements_ptr_;
                }
                /// Returns the expected mean
                double expected_mean() const { return number_generator.mean(NPOINTS); }
                /// Returns the expected error if uncorrelated
                double expected_uncorr_err() const { return number_generator.error(NPOINTS); }
            };

            /// Class to generate a single accumulator A with correlated data of type T
            // FXIME: is it really needed? Is it not the same as AccResultGenerator<...,CorrelatedData>?
            template <template<typename> class A, typename T, std::size_t NPOINTS_P=1000, std::size_t CORRL_P=10, unsigned int VSZ_P=3, typename NG=RandomData>
            class acc_one_correlated_gen {
              private:
                alps::accumulators::result_set* results_ptr_;
                alps::accumulators::accumulator_set* measurements_ptr_;
                CorrelatedData<CORRL_P,NG> number_generator;

              public:
                typedef CorrelatedData<CORRL_P,NG> number_generator_type;
                typedef T data_type;
                typedef A<T> acc_type;

                static const std::size_t NPOINTS=NPOINTS_P; /// < Number of data points
                static const unsigned int VSIZE=VSZ_P; /// size of the vector
                static const std::size_t CORRL=CORRL_P; /// correlation length (samples)

                const std::string acc_name;

              private:
                /// Generate the data points for the accumulator
                void init_() {
                    measurements_ptr_=new alps::accumulators::accumulator_set();
                    alps::accumulators::accumulator_set& m=*measurements_ptr_;
                    m << acc_type(acc_name);

                    for (size_t i=0; i<NPOINTS; ++i) {
                        data_type sample=gen_data<data_type>(number_generator(), VSIZE);
                        m[acc_name]  << sample;
                    }
                    results_ptr_=new alps::accumulators::result_set(m);
                }

              public:
                /// Free the memory allocated in the constructor
                virtual ~acc_one_correlated_gen()
                {
                    delete results_ptr_;
                    delete measurements_ptr_;
                }

                /// Generate the data points for the accumulator, seed number generator with a default value
                acc_one_correlated_gen(const std::string& name="data")
                    : acc_name(name)
                { init_(); }

                /// Generate the data points for the accumulator, seed number generator with a given value
                acc_one_correlated_gen(double ini, const std::string& name="data")
                    : number_generator(ini), acc_name(name)
                { init_(); }

                /// Returns extracted result set
                const alps::accumulators::result_set& results() const
                {
                    return *results_ptr_;
                }

                /// Returns extracted result set as non-const
                alps::accumulators::result_set& results()
                {
                    return *results_ptr_;
                }

                /// Returns the accumulator set
                const alps::accumulators::accumulator_set& accumulators() const
                {
                    return *measurements_ptr_;
                }

                /// Returns the accumulator set as non-const
                alps::accumulators::accumulator_set& accumulators()
                {
                    return *measurements_ptr_;
                }

                /// Returns extracted result
                const alps::accumulators::result_wrapper& result() const
                {
                    return results()[acc_name];
                }

                /// Returns the accumulator set
                const alps::accumulators::accumulator_wrapper& accumulator() const
                {
                    return accumulators()[acc_name];
                }

                /// Returns extracted result as non-const
                alps::accumulators::result_wrapper& result()
                {
                    return results()[acc_name];
                }

                /// Returns the accumulator set as non-const
                alps::accumulators::accumulator_wrapper& accumulator()
                {
                    return accumulators()[acc_name];
                }

                /// Returns the expected mean
                double expected_mean() const { return number_generator.mean(NPOINTS); }
                /// Returns the expected error if uncorrelated
                double expected_uncorr_err() const { return number_generator.error(NPOINTS); }
            };

        } // tesing::
    } // accumulators::
} // alps::


#endif /* ALPS_ACCUMULATORS_TESTS_ACCUMULATOR_GENERATOR_HPP_INCLUDED */
