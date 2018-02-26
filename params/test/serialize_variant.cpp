/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file serialize_variant.cpp
    
    @brief Tests memory dumping/undumping of boost::variant

    @note These are imlementation details as of now

    @note The disabled tests are meant to fail: the POD
    producer/consumer is limited by design.
*/

#include <alps/params/serialize_variant.hpp>
#include <alps/params/dict_types.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

#include "./dict_values_test.hpp"

/// Consumer class to dump a POD object to memory
struct pod_mem_consumer {
    const void* buf;
    std::size_t size;

    pod_mem_consumer() : buf(), size() {}

    template <typename T>
    void operator()(const T& val) {
        buf=&val;
        size=sizeof(val);
    }
};

/// Producer class to recreate a POD object from memory
struct pod_mem_producer {
    const void* buf;
    std::size_t size;
    int target_which;
    int which_count;

    pod_mem_producer(const void* b, std::size_t s, int which)
        : buf(b), size(s), target_which(which), which_count(0)
    {}

    template <typename T>
    boost::optional<T> operator()(const T*)
    {
        boost::optional<T> ret;
        if (target_which==which_count) {
            if (sizeof(T)!=size) throw std::invalid_argument("Size mismatch");
            ret=*static_cast<const T*>(buf);
        }
        ++which_count;
        return ret;
    }
};


typedef alps::detail::variant_serializer<alps::params_ns::detail::dict_all_types,
                                         pod_mem_consumer, pod_mem_producer> var_serializer;
typedef var_serializer::variant_type variant_type;

namespace aptest=alps::params_ns::testing;
namespace apd=alps::params_ns::detail;

inline bool operator==(const apd::None&, const apd::None&) { return true; }

// parameterized over bound type
template <typename T>
class VarSerialTest : public ::testing::Test {
    T val1_, val2_;
  public:
    VarSerialTest()
        : val1_(aptest::data_trait<T>::get(true)),
          val2_(aptest::data_trait<T>::get(false))
    { }

    void sameScope()
    {
        variant_type var1;
        var1=val1_;
        pod_mem_consumer consumer;
        var_serializer::consume(consumer, var1);

        pod_mem_producer producer(consumer.buf, consumer.size, var1.which());
        
        variant_type var2=var_serializer::produce(producer);

        ASSERT_EQ(var1.which(), var2.which());
        T actual=boost::get<T>(var2);
        EXPECT_EQ(val1_, actual);
    }

    void copyData()
    {
        int which1=-1;
        typedef unsigned char byte;
        typedef std::vector<byte> raw_mem;

        raw_mem buf;

        {
            variant_type var1;
            var1=val1_;
            which1=var1.which();
            
            pod_mem_consumer consumer;
            var_serializer::consume(consumer, var1);

            buf.insert(buf.end(),
                       static_cast<const byte*>(consumer.buf),
                       static_cast<const byte*>(consumer.buf)+consumer.size);
        }

        pod_mem_producer producer(&buf[0], buf.size(), which1);
        variant_type var2=var_serializer::produce(producer);

        ASSERT_EQ(which1, var2.which());
        T actual=boost::get<T>(var2);
        EXPECT_EQ(val1_, actual);
    }
};

typedef ::testing::Types<
    bool
    ,
    int
    ,
    long
    ,
    unsigned long
    ,
    double
    /* The following types fail: the producer/consumer handles only PODs */
    // ,
    // std::string
    // ,
    // std::vector<int>
    // ,
    // std::pair<std::string, int>
    > MyTypes;

TYPED_TEST_CASE(VarSerialTest, MyTypes);

TYPED_TEST(VarSerialTest, sameScope) { this->sameScope(); }
TYPED_TEST(VarSerialTest, copyData) { this->copyData(); }
