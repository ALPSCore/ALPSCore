/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Generate a parameter object with a valid value in a various ways, test conformance to specifications */

#include <boost/lexical_cast.hpp>

#include "alps/params.hpp"
#include "gtest/gtest.h"
#include "param_generators_v2.hpp"

#include "alps/utilities/temporary_filename.hpp"
#include "alps/hdf5/archive.hpp"

using namespace alps::params_ns::testing;

// GoogleTest fixture: parametrized by parameter generator type
template <typename G>
class ValidParamTest : public ::testing::Test {
    typedef G generator_type;
    generator_type gen;
    typedef typename G::value_type value_type;
    bool choice; ///< the choice of value provided by generator

    static const bool has_larger_type=G::data_trait_type::has_larger_type;
    static const bool has_smaller_type=G::data_trait_type::has_smaller_type;

    typedef typename G::data_trait_type::larger_type larger_type;
    typedef typename G::data_trait_type::smaller_type smaller_type;
    typedef typename G::data_trait_type::incompatible_type incompatible_type;

    public:

    ValidParamTest(): choice(gen.choice==value_choice::CONSTRUCTION)
    {
        // Sanity check: the generator must generate the parameter with a definite associated value
        if (gen.choice!=value_choice::CONSTRUCTION && gen.choice!=value_choice::DEFINITION) {
            throw std::logic_error("This test fixture must use generator wth a definite associated value");
        }
    }
    
    // Explicit cast to the same type: should return value
    void explicit_cast()
    {
        value_type x=gen.param["param"].template as<value_type>();
        EXPECT_EQ(gen.expected_value, x);
    }

    // Untyped existence check
    void exists()
    {
        EXPECT_TRUE(gen.param.exists("param"));
    }

    // Typed existence check
    void exists_same()
    {
        EXPECT_TRUE(gen.param.template exists<value_type>("param"));
    }

    // Implicit cast to the same type: should return value
    void implicit_cast()
    {
        value_type x=gen.param["param"];
        EXPECT_EQ(gen.expected_value, x);
    }

    // Explicit cast to a larger type: should return value
    void explicit_cast_to_larger()
    {
        if (!has_larger_type) return;
        larger_type x=gen.param["param"].template as<larger_type>();
        EXPECT_EQ(gen.expected_value, x);
    }

    // Implicit cast to a larger type: should return value
    void implicit_cast_to_larger()
    {
        if (!has_larger_type) return;
        larger_type x=gen.param["param"];
        EXPECT_EQ(gen.expected_value, x);
    }

    // Existence check with a larger type: should return true
    void exists_larger()
    {
        if (!has_larger_type) return;
        EXPECT_TRUE(gen.param.template exists<larger_type>("param"));
    }

    // Explicit cast to a smaller type: ?? throw
    void explicit_cast_to_smaller()
    {
        if (!has_smaller_type) return;
        EXPECT_ANY_THROW(gen.param["param"].template as<smaller_type>()); // FIXME: fix throw
    }

    // Implicit cast to a smaller type: ?? throw
    void implicit_cast_to_smaller()
    {
        if (!has_smaller_type) return;
        EXPECT_ANY_THROW(smaller_type x=gen.param["param"]);  // FIXME: fix throw
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
        EXPECT_ANY_THROW(gen.param["param"].template as<incompatible_type>()); // FIXME: fix throw
    }

    // Implicit cast to an incompatible type: throw
    void implicit_cast_to_incompatible()
    {
        EXPECT_ANY_THROW(incompatible_type x=gen.param["param"]);  // FIXME: fix throw
    }

    // Existence check with an incompatible type: should return false
    void exists_incompatible()
    {
        EXPECT_FALSE(gen.param.template exists<incompatible_type>("param"));
    }

    // Accessing non-existing name, const object: throw on access
    void access_nonexistent_const()
    {
        const alps::params& cpar=gen.param;
        EXPECT_ANY_THROW(cpar["nosuchparam"]); // FIXME: fix throw
    }

    // Accessing non-existing name, non-const object: allow
    void access_nonexistent()
    {
        gen.param["nosuchparam"];
    }

    // Assigning from non-existing name, non-const object:  throw
    void assign_nonexistent()
    {
        EXPECT_ANY_THROW(value_type x=gen.param["nosuchparam"]); // FIXME: fix throw
    }
    
    // Existence check for non-exisiting: should return false
    void exists_nonexistent()
    {
        EXPECT_FALSE(gen.param.exists("nosuchparam"));
    }

    // Typed existence check for non-exisiting: should return false
    void exists_nonexistent_typed()
    {
        EXPECT_FALSE(gen.param.template exists<value_type>("nosuchparam"));
    }

    // Assignment from the same type: should acquire value
    void assign_same_type()
    {
        value_type v=data_trait<value_type>::get(!choice); // get the value other than the one associated with the parameter
        gen.param["param"]=v;
        EXPECT_EQ(v, gen.param["param"]);
    }

    // Assignment from a smaller type: should preserve type, acquire value
    void assign_smaller_type()
    {
        if (!has_smaller_type) return;
        smaller_type x=data_trait<smaller_type>::get(!choice);
        gen.param["param"]=x;

        value_type v=gen.param["param"];
        EXPECT_EQ(x, v);
    }

    // Assignment from larger type: should preserve type, reject value
    void assign_larger_type()
    {
        if (!has_larger_type) return;
        larger_type x=data_trait<larger_type>::get(!choice);
        
        EXPECT_ANY_THROW(gen.param["param"]=x); // FIXME: fix throw

        value_type v=gen.param["param"];
        EXPECT_EQ(gen.expected_value, v);
    }

    // Assignment from an incompatible type: should preserve type, reject value
    void assign_incompatible_type()
    {
        incompatible_type x=data_trait<incompatible_type>::get(!choice);
        
        EXPECT_ANY_THROW(gen.param["param"]=x); // FIXME: fix throw

        value_type v=gen.param["param"];
        EXPECT_EQ(gen.expected_value, v);
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

    // define()-ness test
    void defined_nonexistent()
    {
        EXPECT_FALSE(gen.param.defined("nosuchparam"));
    }

    // Saving to and restoring from archive: preserve value
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

        value_type v=p2["param"];
        EXPECT_EQ(gen.expected_value, v);
    }

// Broadcasting: preserve value // FIXME: not implemented
};


TYPED_TEST_CASE_P(ValidParamTest);

TYPED_TEST_P(ValidParamTest,ExplicitCast ) { this->explicit_cast(); }
TYPED_TEST_P(ValidParamTest,Exists) { this->exists(); }
TYPED_TEST_P(ValidParamTest,ExistsSameT) { this->exists_same(); }
TYPED_TEST_P(ValidParamTest,ImplicitCast ) { this->implicit_cast(); }
TYPED_TEST_P(ValidParamTest,ExplicitCastToLarger ) { this->explicit_cast_to_larger(); }
TYPED_TEST_P(ValidParamTest,ImplicitCastToLarger ) { this->implicit_cast_to_larger(); }
TYPED_TEST_P(ValidParamTest,ExistsLarger) { this->exists_larger(); }
TYPED_TEST_P(ValidParamTest,ExplicitCastToSmaller ) { this->explicit_cast_to_smaller(); }
TYPED_TEST_P(ValidParamTest,ImplicitCastToSmaller ) { this->implicit_cast_to_smaller(); }
TYPED_TEST_P(ValidParamTest,ExistsSmaller) { this->exists_smaller(); }
TYPED_TEST_P(ValidParamTest,ExplicitCastToIncompatible ) { this->explicit_cast_to_incompatible(); }
TYPED_TEST_P(ValidParamTest,ImplicitCastToIncompatible ) { this->implicit_cast_to_incompatible(); }
TYPED_TEST_P(ValidParamTest,ExistsIncompatible) { this->exists_incompatible(); }
TYPED_TEST_P(ValidParamTest,AccessNonexistentConst ) { this->access_nonexistent_const(); }
TYPED_TEST_P(ValidParamTest,AccessNonexistent ) { this->access_nonexistent(); }
TYPED_TEST_P(ValidParamTest,AssignNonexistent ) { this->assign_nonexistent(); }
TYPED_TEST_P(ValidParamTest,ExistsNonexistent) { this->exists_nonexistent(); }
TYPED_TEST_P(ValidParamTest,ExistsNonexistentTyped) { this->exists_nonexistent_typed(); }
TYPED_TEST_P(ValidParamTest,AssignSameType ) { this->assign_same_type(); }
TYPED_TEST_P(ValidParamTest,AssignSmallerType ) { this->assign_smaller_type(); }
TYPED_TEST_P(ValidParamTest,AssignLargerType ) { this->assign_larger_type(); }
TYPED_TEST_P(ValidParamTest,AssignIncompatibleType ) { this->assign_incompatible_type(); }
TYPED_TEST_P(ValidParamTest,RedefineSameType ) { this->redefine_same_type(); }
TYPED_TEST_P(ValidParamTest,RedefineAnotherType ) { this->redefine_another_type(); }
TYPED_TEST_P(ValidParamTest,Defined) { this->defined(); }
TYPED_TEST_P(ValidParamTest,DefinedNonexistent) { this->defined_nonexistent(); }
TYPED_TEST_P(ValidParamTest,SaveRestore ) { this->save_restore(); }

REGISTER_TYPED_TEST_CASE_P(ValidParamTest,
                           ExplicitCast,
                           Exists,
                           ExistsSameT,
                           ImplicitCast,
                           ExplicitCastToLarger,
                           ImplicitCastToLarger,
                           ExistsLarger,
                           ExplicitCastToSmaller,
                           ImplicitCastToSmaller,
                           ExistsSmaller,
                           ExplicitCastToIncompatible,
                           ImplicitCastToIncompatible,
                           ExistsIncompatible,
                           AccessNonexistentConst,
                           AccessNonexistent,
                           AssignNonexistent,
                           ExistsNonexistent,
                           ExistsNonexistentTyped,
                           AssignSameType,
                           AssignSmallerType,
                           AssignLargerType,
                           AssignIncompatibleType,
                           RedefineSameType,
                           RedefineAnotherType,
                           Defined,
                           DefinedNonexistent,
                           SaveRestore);


/* A defined-value parameter can be created by:
   1) [ default, parsing missing, parsing existing ] x [ assignment, defining with default]
   2) [ parsing existing ] x [ defining without default ]
   3) copy (not tested here)
   4) load (not tested here)
*/


/*
  Here is the helper `sh` script to generate test cases::
  
  for type in bool char int long double; do \
  echo "typedef ::testing::Types<"; \
  for cons in DefaultConstruct MissingCmdlineConstruct PresentCmdlineConstruct; do \
  for def  in AssignDefine DefaultDefine; do \
  expected='DEFINITION'; \
  [ $cons = PresentCmdlineConstruct ] && [ $def = DefaultDefine ] && expected='CONSTRUCTION'; \
  echo "ParamGenerator<$type, $cons, $def, value_choice::$expected>,"; \
  done; done; \
  echo "ParamGenerator<$type, PresentCmdlineConstruct, NoDefaultDefine, value_choice::CONSTRUCTION>"; \
  typename=${type/ /_}_params_generators; \
  echo "> $typename;"; echo; \
  echo "INSTANTIATE_TYPED_TEST_CASE_P(${type/ }ValidParamTest, ValidParamTest, $typename);"; echo; \
  done
*/

typedef ::testing::Types<
    ParamGenerator<bool, DefaultConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<bool, DefaultConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<bool, MissingCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<bool, MissingCmdlineConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<bool, PresentCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<bool, PresentCmdlineConstruct, DefaultDefine, value_choice::CONSTRUCTION>,
    ParamGenerator<bool, PresentCmdlineConstruct, NoDefaultDefine, value_choice::CONSTRUCTION>
    > bool_params_generators;

INSTANTIATE_TYPED_TEST_CASE_P(boolValidParamTest, ValidParamTest, bool_params_generators);

typedef ::testing::Types<
    ParamGenerator<char, DefaultConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<char, DefaultConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<char, MissingCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<char, MissingCmdlineConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<char, PresentCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<char, PresentCmdlineConstruct, DefaultDefine, value_choice::CONSTRUCTION>,
    ParamGenerator<char, PresentCmdlineConstruct, NoDefaultDefine, value_choice::CONSTRUCTION>
    > char_params_generators;

INSTANTIATE_TYPED_TEST_CASE_P(charValidParamTest, ValidParamTest, char_params_generators);

typedef ::testing::Types<
    ParamGenerator<int, DefaultConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<int, DefaultConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<int, MissingCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<int, MissingCmdlineConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<int, PresentCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<int, PresentCmdlineConstruct, DefaultDefine, value_choice::CONSTRUCTION>,
    ParamGenerator<int, PresentCmdlineConstruct, NoDefaultDefine, value_choice::CONSTRUCTION>
    > int_params_generators;

INSTANTIATE_TYPED_TEST_CASE_P(intValidParamTest, ValidParamTest, int_params_generators);

typedef ::testing::Types<
    ParamGenerator<long, DefaultConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<long, DefaultConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<long, MissingCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<long, MissingCmdlineConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<long, PresentCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<long, PresentCmdlineConstruct, DefaultDefine, value_choice::CONSTRUCTION>,
    ParamGenerator<long, PresentCmdlineConstruct, NoDefaultDefine, value_choice::CONSTRUCTION>
    > long_params_generators;

INSTANTIATE_TYPED_TEST_CASE_P(longValidParamTest, ValidParamTest, long_params_generators);

typedef ::testing::Types<
    ParamGenerator<double, DefaultConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<double, DefaultConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<double, MissingCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<double, MissingCmdlineConstruct, DefaultDefine, value_choice::DEFINITION>,
    ParamGenerator<double, PresentCmdlineConstruct, AssignDefine, value_choice::DEFINITION>,
    ParamGenerator<double, PresentCmdlineConstruct, DefaultDefine, value_choice::CONSTRUCTION>,
    ParamGenerator<double, PresentCmdlineConstruct, NoDefaultDefine, value_choice::CONSTRUCTION>
    > double_params_generators;

INSTANTIATE_TYPED_TEST_CASE_P(doubleValidParamTest, ValidParamTest, double_params_generators);

