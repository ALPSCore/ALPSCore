/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Generate a parameter object without a valid parameter value in a various ways, test conformance to specifications */

#include <boost/lexical_cast.hpp>

#include "alps/params.hpp"
#include "gtest/gtest.h"
#include "param_generators_v2.hpp"

#include "alps/utilities/temporary_filename.hpp"
#include "alps/hdf5/archive.hpp"

using namespace alps::params_ns::testing;

// GoogleTest fixture: parametrized by parameter generator type
template <typename G>
class UndefParamTest : public ::testing::Test {
    typedef G generator_type;
    generator_type gen;
    typedef typename G::value_type value_type;

    static const bool has_larger_type=G::data_trait_type::has_larger_type;
    static const bool has_smaller_type=G::data_trait_type::has_smaller_type;

    typedef typename G::data_trait_type::larger_type larger_type;
    typedef typename G::data_trait_type::smaller_type smaller_type;
    typedef typename G::data_trait_type::incompatible_type incompatible_type;

    public:

    UndefParamTest()
    {
        // Sanity check: the generator must generate the parameter without a definite associated value
        if (gen.choice!=value_choice::UNDEFINED) {
            throw std::logic_error("This test fixture must use generator with an undefined associated value");
        }
    }
    
    // Explicit cast to the same type: should throw
    void explicit_cast()
    {
        EXPECT_ANY_THROW(gen.param["param"].template as<value_type>()); // FIXME: exception type
    }

    // Untyped existence check
    void exists()
    {
        EXPECT_FALSE(gen.param.exists("param"));
    }

    // Typed existence check
    void exists_same()
    {
        EXPECT_FALSE(gen.param.template exists<value_type>("param"));
    }

    // Implicit cast to the same type: should throw on assignment
    void implicit_cast()
    {
        EXPECT_ANY_THROW(value_type x=gen.param["param"]);
    }

    // Accessing const object: throw on access
    void access_const()
    {
        const alps::params& cpar=gen.param;
        EXPECT_ANY_THROW(cpar["param"]); // FIXME: fix throw
    }

    // Accessing non-const object: allow
    void access_nonexistent()
    {
        gen.param["param"];
    }

    // Assignment from the same type: should acquire value
    void assign_same_type()
    {
        value_type v=data_trait<value_type>::get(true); // get some value
        gen.param["param"]=v;
        EXPECT_EQ(v, gen.param["param"]);
    }

    // Assignment from a smaller type: should preserve type, acquire value
    void assign_smaller_type()
    {
        if (!has_smaller_type) return;
        smaller_type x=data_trait<smaller_type>::get(true);
        gen.param["param"]=x;

        value_type v=gen.param["param"];
        EXPECT_EQ(x, v);
    }

    // Assignment from larger type: should preserve type, reject value
    void assign_larger_type()
    {
        if (!has_larger_type) return;
        larger_type x=data_trait<larger_type>::get(true);
        
        EXPECT_ANY_THROW(gen.param["param"]=x); // FIXME: fix throw

        EXPECT_ANY_THROW(value_type v=gen.param["param"]); // FIXME: fix throw
    }

    // Assignment from an incompatible type: should preserve type, reject value
    void assign_incompatible_type()
    {
        incompatible_type x=data_trait<incompatible_type>::get(true);
        
        EXPECT_ANY_THROW(gen.param["param"]=x); // FIXME: fix throw

        EXPECT_ANY_THROW(value_type v=gen.param["param"]); // FIXME: fix throw
    }


    // define() for the same type: throw
    void redefine_same_type()
    {
        EXPECT_ANY_THROW(gen.param.template define<value_type>("param", "Redifinition of a parameter")); // FIXME: fix throw
    }

    // define() for another type: throw
    void redefine_another_type()
    {
        EXPECT_ANY_THROW(gen.param.template define<incompatible_type>("param", "Redifinition of a parameter")); // FIXME: fix throw
    }

    // define()-ness test
    void defined()
    {
        EXPECT_TRUE(gen.param.defined("param"));
    }

    // Saving to and restoring from archive: preserve "defined with no value" status
    void save_restore()
    {
        // Save to archive
        std::string filename(alps::temporary_filename("hdf5_file")+".h5");
        {
            alps::hdf5::archive oar(filename, "w");
            gen.param.save(oar);
        }

        // Load from archive
        alps::params p2;
        {
            alps::hdf5::archive iar(filename, "r");
            p2.load(iar);
        }

        EXPECT_TRUE(p2.defined("param"));
        EXPECT_ANY_THROW(value_type v=p2["param"]);
    }

// Broadcasting: preserve value // FIXME: not implemented
};


TYPED_TEST_CASE_P(UndefParamTest);

TYPED_TEST_P(UndefParamTest,ExplicitCast ) { this->explicit_cast(); }
TYPED_TEST_P(UndefParamTest,Exists) { this->exists(); }
TYPED_TEST_P(UndefParamTest,ExistsSameT) { this->exists_same(); }
TYPED_TEST_P(UndefParamTest,ImplicitCast ) { this->implicit_cast(); }
TYPED_TEST_P(UndefParamTest,AccessConst ) { this->access_const(); }
TYPED_TEST_P(UndefParamTest,AccessNonexistent ) { this->access_nonexistent(); }
TYPED_TEST_P(UndefParamTest,AssignSameType ) { this->assign_same_type(); }
TYPED_TEST_P(UndefParamTest,AssignSmallerType ) { this->assign_smaller_type(); }
TYPED_TEST_P(UndefParamTest,AssignLargerType ) { this->assign_larger_type(); }
TYPED_TEST_P(UndefParamTest,AssignIncompatibleType ) { this->assign_incompatible_type(); }
TYPED_TEST_P(UndefParamTest,RedefineSameType ) { this->redefine_same_type(); }
TYPED_TEST_P(UndefParamTest,RedefineAnotherType ) { this->redefine_another_type(); }
TYPED_TEST_P(UndefParamTest,Defined) { this->defined(); }
TYPED_TEST_P(UndefParamTest,SaveRestore ) { this->save_restore(); }

REGISTER_TYPED_TEST_CASE_P(UndefParamTest,
                           ExplicitCast,
                           Exists,
                           ExistsSameT,
                           ImplicitCast,
                           AccessConst,
                           AccessNonexistent,
                           AssignSameType,
                           AssignSmallerType,
                           AssignLargerType,
                           AssignIncompatibleType,
                           RedefineSameType,
                           RedefineAnotherType,
                           Defined,
                           SaveRestore);


/*
  for type in bool char int long double; do
  echo "typedef ::testing::Types<ALPS_PARAMS_TEST_NxParamTestTypes($type)> ${type}_Params_Generators;";
  echo "INSTANTIATE_TYPED_TEST_CASE_P(${type}ParamTest, UndefParamTest, ${type}_Params_Generators);";
  done
*/

typedef ::testing::Types<ALPS_PARAMS_TEST_NxParamTestTypes(bool)> bool_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(boolParamTest, UndefParamTest, bool_Params_Generators);
typedef ::testing::Types<ALPS_PARAMS_TEST_NxParamTestTypes(char)> char_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(charParamTest, UndefParamTest, char_Params_Generators);
typedef ::testing::Types<ALPS_PARAMS_TEST_NxParamTestTypes(int)> int_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(intParamTest, UndefParamTest, int_Params_Generators);
typedef ::testing::Types<ALPS_PARAMS_TEST_NxParamTestTypes(long)> long_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(longParamTest, UndefParamTest, long_Params_Generators);
typedef ::testing::Types<ALPS_PARAMS_TEST_NxParamTestTypes(double)> double_Params_Generators;
INSTANTIATE_TYPED_TEST_CASE_P(doubleParamTest, UndefParamTest, double_Params_Generators);
