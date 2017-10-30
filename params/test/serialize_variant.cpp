/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file serialize_variant.cpp
    
    @brief Tests memory dumping/undumping of boost::variant

    @note These are imlementation details as of now 
*/

#include <alps/params/serialize_variant.hpp>
#include <alps/params/dict_types.hpp>
#include <gtest/gtest.h>
#include <vector>

#include "./dict_values_test.hpp"

typedef alps::detail::variant_serializer<alps::params_ns::detail::dict_all_types> var_serializer;
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
        var_serializer::variant_mem_view view=var_serializer::to_view(var1);
        
        variant_type var2=var_serializer::from_view(view);

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
            
            var_serializer::variant_mem_view view=var_serializer::to_view(var1);

            buf.insert(buf.end(),
                       static_cast<const byte*>(view.buf),
                       static_cast<const byte*>(view.buf)+view.size);
        }

        var_serializer::variant_mem_view view2(which1, &buf[0], buf.size());
        
        variant_type var2=var_serializer::from_view(view2);

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
    ,
    std::string
    ,
    std::vector<int>
    ,
    std::pair<std::string, int>
    > MyTypes;

TYPED_TEST_CASE(VarSerialTest, MyTypes);

TYPED_TEST(VarSerialTest, sameScope) { this->sameScope(); }
TYPED_TEST(VarSerialTest, copyData) { this->copyData(); }
