/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file custom_type.cpp: Test for using a custom type */

// For remove()
#include <cstdio>

// Must be included before accumulators.hpp
#include "custom_type.hpp"

// Defines the notion of accumulator, instantiates templates defined above.
#include "alps/accumulators.hpp"


#include "gtest/gtest.h"
#include "alps/utilities/gtest_par_xml_output.hpp"
#include "accumulator_generator.hpp"


// Data generation functions

namespace alps {
    namespace accumulators {
        namespace testing {

            /// Generate a value of `my_custom_type<T>`
            template <typename T>
            struct gen_data< my_custom_type<T> > {
                typedef my_custom_type<T> value_type;
                T val_;
                gen_data(T val, unsigned int =0) : val_(val) {}
                operator value_type() const { return value(); }
                value_type value() const { return value_type::generate(val_); }
            };
        
    
            // Data inquiry functions
            template <typename T>
            double get_value(const T& x) {
                return x;
            }
            
            template <typename T>
            double get_value(const my_custom_type<T>& x) {
                return x.get_my_value();
            }
        } // testing::
    } // accumulators::
} // alps::


using namespace alps::accumulators::testing;

// to pass accumulator types
template <template<typename> class A, typename T>
struct AccumulatorTypeGenerator {
    typedef acc_one_correlated_gen<A,T,     /*NPOINTS*/50000,/*CORRL*/5,/*VLEN*/0,RandomData>   accumulator_gen_type;     // generator for A<T>
    typedef acc_one_correlated_gen<A,double,/*NPOINTS*/50000,/*CORRL*/0 ,/*VLEN*/0,ConstantData> dbl_accumulator_gen_type; // generator for A<double>
};

template <typename G>
struct CustomTypeAccumulatorTest : public testing::Test {
    typedef typename G::accumulator_gen_type acc_gen_type;
    typedef typename acc_gen_type::acc_type acc_type;
    typedef typename acc_type::accumulator_type raw_acc_type;
    typedef typename acc_type::result_type raw_result_type;
    typedef typename alps::accumulators::value_type<raw_acc_type>::type value_type;
    
    typedef typename G::dbl_accumulator_gen_type dbl_acc_gen_type;
    typedef typename dbl_acc_gen_type::acc_type dbl_acc_type;

    static const std::size_t NPOINTS=acc_gen_type::NPOINTS; 
    
    acc_gen_type acc_gen;
    dbl_acc_gen_type dbl_acc_gen;

    // Ugly, but should work
    static const bool is_mean_acc=boost::is_same<alps::accumulators::MeanAccumulator<value_type>,
                                                 acc_type>::value;
    static const bool is_nobin_acc=boost::is_same<alps::accumulators::NoBinningAccumulator<value_type>,
                                                  acc_type>::value;
    
    static double tol() { return 5.E-3; }

    CustomTypeAccumulatorTest() {
        alps::accumulators::accumulator_set::register_serializable_type<raw_acc_type>();
        alps::accumulators::result_set::register_serializable_type<raw_result_type>();
    }

    void TestH5ScalarType() {
        typedef typename alps::hdf5::scalar_type<value_type>::type stype;
        EXPECT_EQ(typeid(typename value_type::my_constituent_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestNumScalarType() {
        typedef typename alps::numeric::scalar<value_type>::type stype;
        EXPECT_EQ(typeid(typename value_type::my_scalar_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestElementType() {
        typedef typename alps::element_type<value_type>::type stype;
        EXPECT_TRUE(alps::is_sequence<value_type>::value);
        EXPECT_EQ(typeid(typename value_type::my_element_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestCount() {
        EXPECT_EQ(NPOINTS, acc_gen.result().count());
    }

    void TestMean() {
        EXPECT_NEAR(acc_gen.expected_mean(), get_value(acc_gen.result().template mean<value_type>()), tol());
    }

    void TestError() {
        if (is_mean_acc) return;
        acc_gen.result().template error<value_type>();
    }

    void TestTau() {
        if (is_mean_acc || is_nobin_acc) return;
        value_type tau=acc_gen.result().template autocorrelation<value_type>();
        std::cout << "Autocorrelation is " << tau << std::endl;
        // EXPECT_NEAR(24, get_value(acc_gen.result().template autocorrelation<value_type>()),1.0); // << FIXME: what should be the correct value?
    }

    void TestScaleConst() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()*2;
        EXPECT_NEAR(acc_gen.expected_mean()*2, get_value(r.mean<value_type>()), tol());
    }

    void TestScale() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()*dbl_acc_gen.result();
        EXPECT_NEAR(acc_gen.expected_mean()*dbl_acc_gen.expected_mean(), get_value(r.mean<value_type>()), tol());
    }

    void TestAddConst() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()+2;
        EXPECT_NEAR(acc_gen.expected_mean()+2, get_value(r.mean<value_type>()), tol());
    }

    void TestAddEqConst() {
        alps::accumulators::result_wrapper& r=acc_gen.results()["data"];
        r+=2;
        EXPECT_NEAR(acc_gen.expected_mean()+2, get_value(r.mean<value_type>()), tol());
    }

    void TestAdd() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()+acc_gen.result();
        EXPECT_NEAR(acc_gen.expected_mean()*2, get_value(r.mean<value_type>()), tol());
    }

    void TestAddEq() {
        alps::accumulators::result_wrapper& r=acc_gen.results()["data"];
        r+=acc_gen.results()["data"];
        EXPECT_NEAR(acc_gen.expected_mean()*2, get_value(r.mean<value_type>()), tol());
    }

    void TestAddScalar() {
        const alps::accumulators::result_wrapper& r=acc_gen.result()+dbl_acc_gen.result();
        double expected_mean=acc_gen.expected_mean()+dbl_acc_gen.expected_mean();
        EXPECT_NEAR(expected_mean, get_value(r.mean<value_type>()), tol());
    }

    void TestAddEqScalar() {
        alps::accumulators::result_wrapper& r=acc_gen.results()["data"];
        r+=dbl_acc_gen.result();
        double expected_mean=acc_gen.expected_mean()+dbl_acc_gen.expected_mean();
        EXPECT_NEAR(expected_mean, get_value(r.mean<value_type>()), tol());
    }

    void TestSaveAccumulator() {
        const std::string fname="save_acc.h5";
        std::remove(fname.c_str());
        const alps::accumulators::accumulator_set& m=acc_gen.accumulators();
        alps::hdf5::archive ar(fname,"w");
        ar["dataset"] << m;
    }

    void TestSaveResult() {
        const std::string fname="save_res.h5";
        std::remove(fname.c_str());
        const alps::accumulators::result_set& res=acc_gen.results();
        alps::hdf5::archive ar(fname,"w");
        ar["dataset"] << res;
    }

    void TestSaveLoadAccumulator() {
        const std::string fname="saveload_acc.h5";
        std::remove(fname.c_str());
        const alps::accumulators::accumulator_set& m=acc_gen.accumulators();
        {
            alps::hdf5::archive ar(fname,"w");
            ar["dataset"] << m;
        }
        alps::accumulators::accumulator_set m1;
        {
            alps::hdf5::archive ar(fname,"r");
            ar["dataset"] >> m1;
        }
        alps::accumulators::result_set r(m);
        alps::accumulators::result_set r1(m1);
        EXPECT_NEAR(get_value(r["data"].mean<value_type>()),get_value(r1["data"].mean<value_type>()),1E-8);
        if (is_mean_acc) return;
        EXPECT_NEAR(get_value(r["data"].error<value_type>()),get_value(r1["data"].error<value_type>()),1E-8);
        if (is_nobin_acc) return;
        EXPECT_NEAR(get_value(r["data"].autocorrelation<value_type>()),get_value(r1["data"].autocorrelation<value_type>()),1E-8);
    }

    void TestSaveLoadResult() {
        const std::string fname="saveload_res.h5";
        std::remove(fname.c_str());
        const alps::accumulators::result_set& r=acc_gen.results();
        {
            alps::hdf5::archive ar(fname,"w");
            ar["dataset"] << r;
        }
        alps::accumulators::result_set r1;
        {
            alps::hdf5::archive ar(fname,"r");
            ar["dataset"] >> r1;
        }
        EXPECT_NEAR(get_value(r["data"].mean<value_type>()),get_value(r1["data"].mean<value_type>()),1E-8);
        if (is_mean_acc) return;
        EXPECT_NEAR(get_value(r["data"].error<value_type>()),get_value(r1["data"].error<value_type>()),1E-8);
        if (is_nobin_acc) return;
        EXPECT_NEAR(get_value(r["data"].autocorrelation<value_type>()),get_value(r1["data"].autocorrelation<value_type>()),1E-8);
    }

#ifdef ALPS_HAVE_MPI                   
    void TestMpiMerge() {
        /* The idea is to verify that M processes running N points each
           accumulated together approximately the same data as a single process
           running M*N points, restarting every N points.
        */
        namespace aa=alps::accumulators;
        boost::mpi::communicator comm;

        aa::accumulator_set total_mset;
        total_mset << acc_type("data");
        aa::accumulator_wrapper& total_acc=total_mset["data"];

        for (int p=0; p<comm.size(); ++p) {
            typename acc_gen_type::number_generator_type number_generator;
            
            for (std::size_t i=0; i<NPOINTS; ++i) {
                total_acc << gen_data<value_type>(number_generator()).value();
            }
        }
        aa::result_set total_rset(total_mset);
        const aa::result_wrapper& rtot=total_rset["data"];

        aa::accumulator_wrapper& aw=acc_gen.accumulator();
        aw.collective_merge(comm,0);
        boost::shared_ptr<aa::result_wrapper> rwptr=aw.result();
        const aa::result_wrapper& rw=*rwptr;

        // FIXME: These tolerances are found by comparison with FullBinningAccumulator<double>,
        //        and are valid for the requested correlation length 5, 50000 points, 2 processes.
        const double mean_tol=1E-8;
        const double err_tol=5E-5;
        const double tau_tol=2.;

        for (int i=0; i<comm.size(); ++i) {
            if (comm.rank() == i
                && i==0 /* FIXME!! See issue #178 */) {
                
                EXPECT_EQ(rtot.count(), rw.count()) << "reported by rank " << i;
                
                EXPECT_NEAR(get_value(rtot.mean<value_type>()),
                            get_value(rw.mean<value_type>()),
                            mean_tol) << "reported by rank " << i;

                if (!is_mean_acc)
                    EXPECT_NEAR(get_value(rtot.error<value_type>()),
                                get_value(rw.error<value_type>()),
                                err_tol) << "reported by rank " << i;
                
                if (!(is_nobin_acc||is_mean_acc)) 
                    EXPECT_NEAR(get_value(rtot.autocorrelation<value_type>()),
                                get_value(rw.autocorrelation<value_type>()),
                                tau_tol) << "reported by rank " << i;
            }
            comm.barrier();
        }
    }
#endif
};

typedef my_custom_type<double> dbl_custom_type;

typedef ::testing::Types<
    AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,dbl_custom_type>
    ,AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator,dbl_custom_type>
    ,AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator, dbl_custom_type>
    ,AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator,      dbl_custom_type>
    // ,AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator, double>
    > MyTypes;

TYPED_TEST_CASE(CustomTypeAccumulatorTest, MyTypes);

#ifdef ALPS_HAVE_MPI

#define RETURN_UNLESS_MASTER if (boost::mpi::communicator().rank()!=0) return
#define MAKE_MPI_TEST(_name_) TYPED_TEST(CustomTypeAccumulatorTest, _name_)  { this->TestFixture::_name_(); }

#else

#define RETURN_UNLESS_MASTER
#define MAKE_MPI_TEST(_name_)

#endif

#define MAKE_TEST(_name_)                                               \
    TYPED_TEST(CustomTypeAccumulatorTest, _name_)  {                    \
        RETURN_UNLESS_MASTER;                                           \
        this->TestFixture::_name_();                                    \
    }


MAKE_TEST(TestH5ScalarType)
MAKE_TEST(TestNumScalarType)
MAKE_TEST(TestElementType)

MAKE_TEST(TestMean)
MAKE_TEST(TestError)
MAKE_TEST(TestTau)

MAKE_TEST(TestScaleConst)
MAKE_TEST(TestScale)
MAKE_TEST(TestAddConst)
MAKE_TEST(TestAdd)
MAKE_TEST(TestAddScalar)
MAKE_TEST(TestAddEqConst)
MAKE_TEST(TestAddEq)
MAKE_TEST(TestAddEqScalar)

MAKE_TEST(TestSaveAccumulator)
MAKE_TEST(TestSaveLoadAccumulator)
MAKE_TEST(TestSaveResult)
MAKE_TEST(TestSaveLoadResult)

MAKE_MPI_TEST(TestMpiMerge)

TEST(CustomTypeAccumulatorTest,saveArray) {
    RETURN_UNLESS_MASTER;
    dbl_custom_type x=gen_data<dbl_custom_type>(1.25);
    std::vector<dbl_custom_type> vx;
    std::vector<double> dx;
    
    for (int i=0; i<10; ++i) {
        vx.push_back(gen_data<dbl_custom_type>(i+0.5));
        dx.push_back(i+0.75);
    }
    std::cout << "x=" << x << "\nvx=" << vx << std::endl;
    {
      alps::hdf5::archive ar("saveload_custom.h5","w");
      std::cout << "Saving custom scalar..." << std::endl;
      ar["single_custom"] << x;
      std::cout << "Saving custom vector..." << std::endl;
      ar["vector_custom"] << vx;
      std::cout << "Saving regular vector..." << std::endl;
      ar["vector_regular"] << dx;
      
    }
    dbl_custom_type y;
    std::vector<dbl_custom_type> vy;
    {
      alps::hdf5::archive ar("saveload_custom.h5","r");
      std::cout << "Loading custom scalar..." << std::endl;
      ar["single_custom"] >> y;
      std::cout << "Loading custom vector..." << std::endl;
      ar["vector_custom"] >> vy;

      my_custom_type<float> fy;
      EXPECT_THROW(ar["single_custom"] >> fy, std::runtime_error);
      
      dbl_custom_type y1;
      EXPECT_THROW(ar["vector_regular"] >> y1, std::runtime_error);
    }
    std::cout << "y=" << y << "\nvy=" << vy << std::endl;
}

#ifdef ALPS_HAVE_MPI
int main(int argc, char** argv)
{
   boost::mpi::environment env(argc, argv, false);
   alps::gtest_par_xml_output tweak;
   tweak(boost::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}    
#else
int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}    
#endif
