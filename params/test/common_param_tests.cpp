/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Generate a parameter object with or without a valid value in a various ways,
    test conformance to specifications */

#include <boost/lexical_cast.hpp>

#include "alps/params.hpp"
#include "gtest/gtest.h"
#include "param_generators_v2.hpp"

#include "alps/utilities/temporary_filename.hpp"
#include "alps/hdf5/archive.hpp"

//FIXME!!! DEBUG!!!
#undef EXPECT_THROW
#define EXPECT_THROW(a,b) a


using namespace alps::params_ns::testing;

// GoogleTest fixture: parametrized by parameter generator type; tests features common for a parameter with or without value
template <typename G>
class AnyParamTest : public ::testing::Test {
    typedef G generator_type;
    generator_type gen;
    typedef typename G::value_type value_type;

    static const bool has_larger_type=G::data_trait_type::has_larger_type;
    static const bool has_smaller_type=G::data_trait_type::has_smaller_type;

    typedef typename G::data_trait_type::larger_type larger_type;
    typedef typename G::data_trait_type::smaller_type smaller_type;
    typedef typename G::data_trait_type::incompatible_type incompatible_type;

    public:

    // Accessing non-const object: always allow
    void access_nonconst()
    {
        gen.param["param"];
    }

    // Explicit cast to a smaller type: throw
    void explicit_cast_to_smaller()
    {
        if (!has_smaller_type) return;
        EXPECT_THROW(gen.param["param"].template as<smaller_type>(), alps::params::type_mismatch);
    }

    // Implicit cast to a smaller type: throw
    void implicit_cast_to_smaller()
    {
        if (!has_smaller_type) return;
        EXPECT_THROW(smaller_type x=gen.param["param"], alps::params::type_mismatch);
    }

    // Existence check with a smaller type: should return false
    void exists_smaller()
    {
        if (!has_smaller_type) return;
        EXPECT_FALSE(gen.param.template exists<smaller_type>("param"));
    }

    // Explicit cast to an incompatible type: throw
    void explicit_cast_to_incompatible()
    {
        EXPECT_THROW(gen.param["param"].template as<incompatible_type>(), alps::params::type_mismatch);
    }

    // Implicit cast to an incompatible type: throw
    void implicit_cast_to_incompatible()
    {
        EXPECT_THROW(incompatible_type x=gen.param["param"], alps::params::type_mismatch);
    }

    // Existence check with an incompatible type: should return false
    void exists_incompatible()
    {
        EXPECT_FALSE(gen.param.template exists<incompatible_type>("param"));
    }

    // define() for the same type: throw
    void redefine_same_type()
    {
        EXPECT_THROW(gen.param.template define<value_type>("param", "Redifinition of a parameter"), alps::params::double_definition);
    }

    // define() for another type: throw
    void redefine_another_type()
    {
        EXPECT_THROW(gen.param.template define<incompatible_type>("param", "Redifinition of a parameter"), alps::params::double_definition);
    }

    // define()-ness test
    void defined()
    {
        EXPECT_TRUE(gen.param.defined("param"));
    }
};


TYPED_TEST_CASE_P(AnyParamTest);

TYPED_TEST_P(AnyParamTest,AccessNonconst) { this->access_nonconst(); }
TYPED_TEST_P(AnyParamTest,ExplicitCastToSmaller) { this->explicit_cast_to_smaller(); }
TYPED_TEST_P(AnyParamTest,ImplicitCastToSmaller) { this->implicit_cast_to_smaller(); }
TYPED_TEST_P(AnyParamTest,ExistsSmaller) { this->exists_smaller(); }
TYPED_TEST_P(AnyParamTest,ExplicitCastToIncompatible) { this->explicit_cast_to_incompatible(); }
TYPED_TEST_P(AnyParamTest,ImplicitCastToIncompatible) { this->implicit_cast_to_incompatible(); }
TYPED_TEST_P(AnyParamTest,ExistsIncompatible) { this->exists_incompatible(); }
TYPED_TEST_P(AnyParamTest,RedefineSameType) { this->redefine_same_type(); }
TYPED_TEST_P(AnyParamTest,RedefineAnotherType) { this->redefine_another_type(); }
TYPED_TEST_P(AnyParamTest,Defined) { this->defined(); }

REGISTER_TYPED_TEST_CASE_P(AnyParamTest,
                           AccessNonconst,
                           ExplicitCastToSmaller,
                           ImplicitCastToSmaller,
                           ExistsSmaller,
                           ExplicitCastToIncompatible,
                           ImplicitCastToIncompatible,
                           ExistsIncompatible,
                           RedefineSameType,
                           RedefineAnotherType,
                           Defined);

/*
  for type in bool char int long double; do
  echo "typedef ::testing::Types<ALPS_PARAMS_TEST_AllParamTestTypes($type)> ${type}_Params_Generators;";
  echo "INSTANTIATE_TYPED_TEST_CASE_P(${type}ParamTest, AnyParamTest, ${type}_Params_Generators);";
  done
*/

typedef ::testing::Types<ALPS_PARAMS_TEST_AllParamTestTypes(bool)> bool_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(boolParamTest, AnyParamTest, bool_Params_Generators);
typedef ::testing::Types<ALPS_PARAMS_TEST_AllParamTestTypes(char)> char_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(charParamTest, AnyParamTest, char_Params_Generators);
typedef ::testing::Types<ALPS_PARAMS_TEST_AllParamTestTypes(int)> int_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(intParamTest, AnyParamTest, int_Params_Generators);
typedef ::testing::Types<ALPS_PARAMS_TEST_AllParamTestTypes(long)> long_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(longParamTest, AnyParamTest, long_Params_Generators);
typedef ::testing::Types<ALPS_PARAMS_TEST_AllParamTestTypes(double)> double_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(doubleParamTest, AnyParamTest, double_Params_Generators);
