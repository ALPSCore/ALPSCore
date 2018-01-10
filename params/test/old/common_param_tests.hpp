/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file common_param_tests.cpp
    Generate a parameter object with in a various ways, test conformance to specifications.
    Header file defining the battery of tests.
*/

#include <boost/lexical_cast.hpp>

#include "alps/params.hpp"
#include "gtest/gtest.h"
#include "param_generators_v2.hpp"

#include "alps/testing/unique_file.hpp"
#include "alps/hdf5/archive.hpp"

using namespace alps::params_ns::testing;


/* param[name]. `name` has a type, therefore it is either assigned or defined (with or without default).
   The `param` object may be obtained by:
   1) Default constructor.
   2) Copy constructor.
   3) Copy.
   4) Load.
   5) Broadcast.
   6) Commandline parsing.
   7) INI-file parsing.

   For (6) and (7), agreement on the names:
   present_def          is present in the command line (with value v1), defined with default (v2);
   missing_def          is missing in the command line, defined with default (v1);
   present_nodef        is present in the command line (with value v1), defined without default;
   assigned             is assigned with value v1;
   missing_nodef        is missing in the command line, defined without default;
   nosuchname           is neither assigned nor defined;

   where v1=get_value<T>(true), v2=get_value<T>(false). 
       
   For (1) (default-constructed), agreement on names:
   "assigned" is assigned with value v1;
   "nosuchname" is neither assigned nor defined.

   (Alternatively, we could have a battery of tests within the same test
   method (or function) that will ensure that the parameter `name`
   behaves as expected. Then we can run those tests on every name and
   parameter that we assigned/modified/changed. The only drawback that
   the tests will not be performed on a fresh object.)
*/
  


// GoogleTest fixture: parametrized by generator type
template <typename G>
class AnyParamTest : public ::testing::Test {
    typedef G generator_type;
    generator_type gen;
    typedef typename G::value_type value_type;
    typedef typename G::data_trait_type data_trait_type;

    static const bool has_larger_type=data_trait_type::has_larger_type;
    static const bool has_smaller_type=data_trait_type::has_smaller_type;

    typedef typename data_trait_type::larger_type larger_type;
    typedef typename data_trait_type::smaller_type smaller_type;
    typedef typename data_trait_type::incompatible_type incompatible_type;

    alps::params& param;
    const value_type val1;
    const value_type val2;
    const std::string expected_origin_name;
    public:

    AnyParamTest():
        param(*gen.param_ptr),
        val1(data_trait_type::get(true)),
        val2(data_trait_type::get(false)),
        expected_origin_name(gen.expected_origin_name())
    { }

    // Accessing const object
    void access_const()
    {
        const alps::params& cpar=param;
        cpar["present_def"];
        cpar["missing_def"];
        cpar["present_nodef"];
        cpar["assigned"];
        EXPECT_THROW(cpar["missing_nodef"], alps::params::uninitialized_value);
        EXPECT_THROW(cpar["nosuchname"], alps::params::uninitialized_value);
    }

    // Accessing non-const object
    void access_nonconst()
    {
        param["present_def"];
        param["missing_def"];
        param["present_nodef"];
        param["assigned"];
        param["missing_nodef"]; // does not throw 
        param["nosuchname"]; // does not throw 
    }

    // Explicit cast to the same type: should return value
    void explicit_cast()
    {
#define as   template as  /* work around a bug (?) in clang, see https://llvm.org/bugs/show_bug.cgi?id=22247 */
        EXPECT_EQ(val1, param["present_def"].as<value_type>());
        EXPECT_EQ(val1, param["missing_def"].as<value_type>());
        EXPECT_EQ(val1, param["present_nodef"].as<value_type>());
        EXPECT_EQ(val1, param["assigned"].as<value_type>());

        EXPECT_THROW(param["missing_nodef"].as<value_type>(), alps::params::uninitialized_value);
        EXPECT_THROW(param["nosuchname"].as<value_type>(), alps::params::uninitialized_value);
#undef as        
    }

    // Untyped existence check
    void exists()
    {
        EXPECT_TRUE(param.exists("present_def"));
        EXPECT_TRUE(param.exists("missing_def"));
        EXPECT_TRUE(param.exists("present_nodef"));
        EXPECT_TRUE(param.exists("assigned"));
        
        EXPECT_FALSE(param.exists("missing_nodef"));
        EXPECT_FALSE(param.exists("nosuchname"));
    }

    // Typed existence check
    void exists_same()
    {
        EXPECT_TRUE(param.exists<value_type>("present_def"));
        EXPECT_TRUE(param.exists<value_type>("missing_def"));
        EXPECT_TRUE(param.exists<value_type>("present_nodef"));
        EXPECT_TRUE(param.exists<value_type>("assigned"));
        
        EXPECT_FALSE(param.exists<value_type>("missing_nodef"));
        EXPECT_FALSE(param.exists<value_type>("nosuchname"));
    }

    // Implicit cast to the same type: should return value
    void implicit_cast()
    {
        {
            value_type x=param["present_def"];
            EXPECT_EQ(val1,x);
        }
        
        {
            value_type x=param["missing_def"];
            EXPECT_EQ(val1,x);
        }

        {
            value_type x=param["present_nodef"];
            EXPECT_EQ(val1,x);
        }

        {
            value_type x=param["assigned"];
            EXPECT_EQ(val1,x);
        }

        {
            EXPECT_THROW(value_type x=param["missing_nodef"], alps::params::uninitialized_value);
        }

        {
            EXPECT_THROW(value_type x=param["nosuchname"], alps::params::uninitialized_value);
        }
    }

    // Implicit assignment to the same type: should assign value
    void implicit_assign()
    {
        {
            value_type x; x=param["present_def"];
            EXPECT_EQ(val1,x);
        }
        
        {
            value_type x; x=param["missing_def"];
            EXPECT_EQ(val1,x);
        }

        {
            value_type x; x=param["present_nodef"];
            EXPECT_EQ(val1,x);
        }

        {
            value_type x; x=param["assigned"];
            EXPECT_EQ(val1,x);
        }

        {
            value_type x;
            EXPECT_THROW(x=param["missing_nodef"], alps::params::uninitialized_value);
        }

        {
            value_type x;
            EXPECT_THROW(x=param["nosuchname"], alps::params::uninitialized_value);
        }
    }

    // Explicit cast to a larger type: should return value
    void explicit_cast_to_larger()
    {
#define as   template as  /* work around a bug (?) in clang, see https://llvm.org/bugs/show_bug.cgi?id=22247 */
        if (!has_larger_type) return;
        EXPECT_EQ(val1, param["present_def"].as<larger_type>());
        EXPECT_EQ(val1, param["missing_def"].as<larger_type>());
        EXPECT_EQ(val1, param["present_nodef"].as<larger_type>());
        EXPECT_EQ(val1, param["assigned"].as<larger_type>());

        EXPECT_THROW(param["missing_nodef"].as<larger_type>(), alps::params::uninitialized_value);
        EXPECT_THROW(param["nosuchname"].as<larger_type>(), alps::params::uninitialized_value);
#undef as        
    }

    // Implicit cast to a larger type: should return value
    void implicit_cast_to_larger()
    {
        if (!has_larger_type) return;
        
        {
            larger_type x=param["present_def"];
            EXPECT_EQ(val1, x);
        }

        {
            larger_type x=param["missing_def"];
            EXPECT_EQ(val1, x);
        }

        {
            larger_type x=param["present_nodef"];
            EXPECT_EQ(val1, x);
        }

        {
            larger_type x=param["assigned"];
            EXPECT_EQ(val1, x);
        }

        {
            EXPECT_THROW(larger_type x=param["missing_nodef"], alps::params::uninitialized_value);
        }

        {
            EXPECT_THROW(larger_type x=param["nosuchname"], alps::params::uninitialized_value);
        }
    }

    // Existence check with a larger type: should return true
    void exists_larger()
    {
        if (!has_larger_type) return;
        EXPECT_TRUE(param.exists<larger_type>("present_def"));
        EXPECT_TRUE(param.exists<larger_type>("missing_def"));
        EXPECT_TRUE(param.exists<larger_type>("present_nodef"));
        EXPECT_TRUE(param.exists<larger_type>("assigned"));
        
        EXPECT_FALSE(param.exists<larger_type>("missing_nodef"));
        EXPECT_FALSE(param.exists<larger_type>("nosuchname"));
    }

    // Explicit cast to a smaller type: ?? throw
    void explicit_cast_to_smaller()
    {
#define as   template as  /* work around a bug (?) in clang, see https://llvm.org/bugs/show_bug.cgi?id=22247 */
        if (!has_smaller_type) return;

        EXPECT_THROW(param["present_def"].as<smaller_type>(),alps::params::type_mismatch);
        EXPECT_THROW(param["missing_def"].as<smaller_type>(),alps::params::type_mismatch);
        EXPECT_THROW(param["present_nodef"].as<smaller_type>(),alps::params::type_mismatch);
        EXPECT_THROW(param["assigned"].as<smaller_type>(),alps::params::type_mismatch);

        EXPECT_THROW(param["missing_nodef"].as<smaller_type>(),alps::params::uninitialized_value);
        EXPECT_THROW(param["nosuchname"].as<smaller_type>(),alps::params::uninitialized_value);
#undef as        
    }

    // Implicit cast to a smaller type: ?? throw
    void implicit_cast_to_smaller()
    {
        if (!has_smaller_type) return;

        EXPECT_THROW(smaller_type x=param["present_def"], alps::params::type_mismatch);
        EXPECT_THROW(smaller_type x=param["missing_def"], alps::params::type_mismatch);
        EXPECT_THROW(smaller_type x=param["present_nodef"], alps::params::type_mismatch);
        EXPECT_THROW(smaller_type x=param["assigned"], alps::params::type_mismatch);

        EXPECT_THROW(smaller_type x=param["missing_nodef"], alps::params::uninitialized_value);
        EXPECT_THROW(smaller_type x=param["nosuchname"], alps::params::uninitialized_value);
    }

    // Existence check with a smaller type: should return false
    void exists_smaller()
    {
        if (!has_smaller_type) return;
        
        EXPECT_FALSE(param.exists<smaller_type>("present_def"));
        EXPECT_FALSE(param.exists<smaller_type>("missing_def"));
        EXPECT_FALSE(param.exists<smaller_type>("present_nodef"));
        EXPECT_FALSE(param.exists<smaller_type>("assigned"));
        EXPECT_FALSE(param.exists<smaller_type>("missing_nodef"));
        EXPECT_FALSE(param.exists<smaller_type>("nosuchname"));
    }

    // Explicit cast to an incompatible type: throw
    void explicit_cast_to_incompatible()
    {
#define as   template as  /* work around a bug (?) in clang, see https://llvm.org/bugs/show_bug.cgi?id=22247 */

        EXPECT_THROW(param["present_def"].as<incompatible_type>(), alps::params::type_mismatch);
        EXPECT_THROW(param["missing_def"].as<incompatible_type>(), alps::params::type_mismatch);
        EXPECT_THROW(param["present_nodef"].as<incompatible_type>(), alps::params::type_mismatch);
        EXPECT_THROW(param["assigned"].as<incompatible_type>(), alps::params::type_mismatch);
        
        EXPECT_THROW(param["missing_nodef"].as<incompatible_type>(), alps::params::uninitialized_value);
        EXPECT_THROW(param["nosuchname"].as<incompatible_type>(), alps::params::uninitialized_value);
#undef as
    }

    // Implicit cast to an incompatible type: throw
    void implicit_cast_to_incompatible()
    {
        EXPECT_THROW(incompatible_type x=param["present_def"], alps::params::type_mismatch);
        EXPECT_THROW(incompatible_type x=param["missing_def"], alps::params::type_mismatch);
        EXPECT_THROW(incompatible_type x=param["present_nodef"], alps::params::type_mismatch);
        EXPECT_THROW(incompatible_type x=param["assigned"], alps::params::type_mismatch);
        
        EXPECT_THROW(incompatible_type x=param["missing_nodef"], alps::params::uninitialized_value);
        EXPECT_THROW(incompatible_type x=param["nosuchname"], alps::params::uninitialized_value);
    }

    // Existence check with an incompatible type: should return false
    void exists_incompatible()
    {
        EXPECT_FALSE(param.exists<incompatible_type>("present_def"));
        EXPECT_FALSE(param.exists<incompatible_type>("missing_def"));
        EXPECT_FALSE(param.exists<incompatible_type>("present_nodef"));
        EXPECT_FALSE(param.exists<incompatible_type>("assigned"));
        EXPECT_FALSE(param.exists<incompatible_type>("missing_nodef"));
        EXPECT_FALSE(param.exists<incompatible_type>("nosuchname"));
    }

    // Assignment from the same type: should acquire value
    void assign_same_type()
    {
        EXPECT_EQ(val1, param["present_def"]);
        param["present_def"]=val2;
        EXPECT_EQ(val2, param["present_def"]);

        EXPECT_EQ(val1, param["missing_def"]);
        param["missing_def"]=val2;
        EXPECT_EQ(val2, param["missing_def"]);

        EXPECT_EQ(val1, param["present_nodef"]);
        param["present_nodef"]=val2;
        EXPECT_EQ(val2, param["present_nodef"]);

        EXPECT_EQ(val1, param["assigned"]);
        param["assigned"]=val2;
        EXPECT_EQ(val2, param["assigned"]);

        param["missing_nodef"]=val2;
        EXPECT_EQ(val2, param["missing_nodef"]);

        param["nosuchname"]=val2;
        EXPECT_EQ(val2, param["nosuchname"]);
    }

    // Assignment from a smaller type: should preserve type if any, acquire value
    void assign_smaller_type()
    {
#define as   template as  /* work around a bug (?) in clang, see https://llvm.org/bugs/show_bug.cgi?id=22247 */

        if (!has_smaller_type) return;
        const value_type v1=data_trait<value_type>::get(true);
        const smaller_type v2=data_trait<smaller_type>::get(false);

        EXPECT_EQ(v1, param["present_def"].as<value_type>());
        param["present_def"]=v2;
        EXPECT_EQ(v2,param["present_def"].as<value_type>());
        EXPECT_FALSE(param.exists<smaller_type>("present_def"));
        
        EXPECT_EQ(v1, param["missing_def"].as<value_type>());
        param["missing_def"]=v2;
        EXPECT_EQ(v2,param["missing_def"].as<value_type>());
        EXPECT_FALSE(param.exists<smaller_type>("missing_def"));
        
        EXPECT_EQ(v1, param["present_nodef"].as<value_type>());
        param["present_nodef"]=v2;
        EXPECT_EQ(v2,param["present_nodef"].as<value_type>());
        EXPECT_FALSE(param.exists<smaller_type>("present_nodef"));
        
        EXPECT_EQ(v1, param["assigned"].as<value_type>());
        param["assigned"]=v2;
        EXPECT_EQ(v2,param["assigned"].as<value_type>());
        EXPECT_FALSE(param.exists<smaller_type>("assigned"));

        // acquires value, but not the type
        param["missing_nodef"]=v2;
        EXPECT_EQ(v2,param["missing_nodef"].as<value_type>());
        EXPECT_FALSE(param.exists<smaller_type>("missing_nodef"));
        
        // acquires value and the type
        param["nosuchname"]=v2;
        EXPECT_EQ(v2,param["nosuchname"].as<smaller_type>());
#undef as
    }

    // Assignment from larger type: should preserve type if any, reject value
    void assign_larger_type()
    {
#define as   template as  /* work around a bug (?) in clang, see https://llvm.org/bugs/show_bug.cgi?id=22247 */

        if (!has_larger_type) return;
        const value_type  v1=data_trait<value_type>::get(true);
        const larger_type v2=data_trait<larger_type>::get(false);

        EXPECT_EQ(v1,param["present_def"].as<value_type>());
        EXPECT_THROW(param["present_def"]=v2, alps::params::type_mismatch);
        EXPECT_EQ(v1,param["present_def"].as<value_type>());
        
        EXPECT_EQ(v1,param["missing_def"].as<value_type>());
        EXPECT_THROW(param["missing_def"]=v2, alps::params::type_mismatch);
        EXPECT_EQ(v1,param["missing_def"].as<value_type>());
        
        EXPECT_EQ(v1,param["present_nodef"].as<value_type>());
        EXPECT_THROW(param["present_nodef"]=v2, alps::params::type_mismatch);
        EXPECT_EQ(v1,param["present_nodef"].as<value_type>());
        
        EXPECT_EQ(v1,param["assigned"].as<value_type>());
        EXPECT_THROW(param["assigned"]=v2, alps::params::type_mismatch);
        EXPECT_EQ(v1,param["assigned"].as<value_type>());

        // Rejects type, does not acquire any value
        EXPECT_THROW(param["missing_nodef"]=v2, alps::params::type_mismatch);
        EXPECT_THROW(param["missing_nodef"].as<value_type>(), alps::params::uninitialized_value);

        // Acquires type and value
        param["nosuchname"]=v2;
        EXPECT_EQ(v2,param["nosuchname"].as<larger_type>());

#undef as        
    }

    // Assignment from an incompatible type: should preserve type, reject value
    void assign_incompatible_type()
    {
#define as   template as  /* work around a bug (?) in clang, see https://llvm.org/bugs/show_bug.cgi?id=22247 */

        const incompatible_type v2=data_trait<incompatible_type>::get(false);

        EXPECT_EQ(val1,param["present_def"].as<value_type>());
        EXPECT_THROW(param["present_def"]=v2, alps::params::type_mismatch);
        EXPECT_EQ(val1,param["present_def"].as<value_type>());
        
        EXPECT_EQ(val1,param["missing_def"].as<value_type>());
        EXPECT_THROW(param["missing_def"]=v2, alps::params::type_mismatch);
        EXPECT_EQ(val1,param["missing_def"].as<value_type>());
        
        EXPECT_EQ(val1,param["present_nodef"].as<value_type>());
        EXPECT_THROW(param["present_nodef"]=v2, alps::params::type_mismatch);
        EXPECT_EQ(val1,param["present_nodef"].as<value_type>());
        
        EXPECT_EQ(val1,param["assigned"].as<value_type>());
        EXPECT_THROW(param["assigned"]=v2, alps::params::type_mismatch);
        EXPECT_EQ(val1,param["assigned"].as<value_type>());

        // Rejects type, does not acquire any value
        EXPECT_THROW(param["missing_nodef"]=v2, alps::params::type_mismatch);
        EXPECT_THROW(param["missing_nodef"].as<value_type>(), alps::params::uninitialized_value);

        // Acquires type and value
        param["nosuchname"]=v2;
        EXPECT_EQ(v2,param["nosuchname"].as<incompatible_type>());
#undef as        
    }

    // define() for the same type: throw
    void redefine_same_type()
    {
        EXPECT_THROW(param.define<value_type>("present_def", "Redefinition"), alps::params::double_definition);
        EXPECT_THROW(param.define<value_type>("missing_def", "Redefinition"), alps::params::double_definition);
        EXPECT_THROW(param.define<value_type>("present_nodef", "Redefinition"), alps::params::double_definition);
        EXPECT_THROW(param.define<value_type>("missing_nodef", "Redefinition"), alps::params::double_definition);
        
        EXPECT_THROW(param.define<value_type>("assigned", "Redefinition"), alps::params::extra_definition);
        
        param.define<value_type>("nosuchname", "Redefinition");
    }


    // define() for another type: throw
    void redefine_another_type()
    {
        EXPECT_THROW(param.define<incompatible_type>("present_def", "Redefinition"), alps::params::double_definition);
        EXPECT_THROW(param.define<incompatible_type>("missing_def", "Redefinition"), alps::params::double_definition);
        EXPECT_THROW(param.define<incompatible_type>("present_nodef", "Redefinition"), alps::params::double_definition);
        EXPECT_THROW(param.define<incompatible_type>("missing_nodef", "Redefinition"), alps::params::double_definition);
        
        EXPECT_THROW(param.define<incompatible_type>("assigned", "Redefinition"), alps::params::extra_definition);
        
        param.define<incompatible_type>("nosuchname", "Redefinition");
    }

    // define()-ness test
    void defined()
    {
        EXPECT_TRUE(param.defined("present_def"));
        EXPECT_TRUE(param.defined("missing_def"));
        EXPECT_TRUE(param.defined("present_nodef"));
        EXPECT_TRUE(param.defined("assigned"));
        EXPECT_TRUE(param.defined("missing_nodef"));
        EXPECT_FALSE(param.defined("nosuchname"));
    }

    // origin name test (the name is, alas, class-specific)
    void origin_name()
    {
        EXPECT_EQ(expected_origin_name,param.get_origin_name());
    }
        

// Saving to and restoring from archive: is a generator
// Copying from another object: is a generator
// Copy-constructing:  is a generator
// Broadcasting: is a generator    
};


TYPED_TEST_CASE_P(AnyParamTest);

TYPED_TEST_P(AnyParamTest,AccessConst) { this->access_const(); }
TYPED_TEST_P(AnyParamTest,AccessNonconst) { this->access_nonconst(); }
TYPED_TEST_P(AnyParamTest,ExplicitCast) { this->explicit_cast(); }
TYPED_TEST_P(AnyParamTest,Exists) { this->exists(); }
TYPED_TEST_P(AnyParamTest,ExistsSame) { this->exists_same(); }
TYPED_TEST_P(AnyParamTest,ImplicitCast) { this->implicit_cast(); }
TYPED_TEST_P(AnyParamTest,ImplicitAssign) { this->implicit_assign(); }
TYPED_TEST_P(AnyParamTest,ExplicitCastToLarger) { this->explicit_cast_to_larger(); }
TYPED_TEST_P(AnyParamTest,ImplicitCastToLarger) { this->implicit_cast_to_larger(); }
TYPED_TEST_P(AnyParamTest,ExistsLarger) { this->exists_larger(); }
TYPED_TEST_P(AnyParamTest,ExplicitCastToSmaller) { this->explicit_cast_to_smaller(); }
TYPED_TEST_P(AnyParamTest,ImplicitCastToSmaller) { this->implicit_cast_to_smaller(); }
TYPED_TEST_P(AnyParamTest,ExistsSmaller) { this->exists_smaller(); }
TYPED_TEST_P(AnyParamTest,ExplicitCastToIncompatible) { this->explicit_cast_to_incompatible(); }
TYPED_TEST_P(AnyParamTest,ImplicitCastToIncompatible) { this->implicit_cast_to_incompatible(); }
TYPED_TEST_P(AnyParamTest,ExistsIncompatible) { this->exists_incompatible(); }
TYPED_TEST_P(AnyParamTest,AssignSameType) { this->assign_same_type(); }
TYPED_TEST_P(AnyParamTest,AssignSmallerType) { this->assign_smaller_type(); }
TYPED_TEST_P(AnyParamTest,AssignLargerType) { this->assign_larger_type(); }
TYPED_TEST_P(AnyParamTest,AssignIncompatibleType) { this->assign_incompatible_type(); }
TYPED_TEST_P(AnyParamTest,RedefineSameType) { this->redefine_same_type(); }
TYPED_TEST_P(AnyParamTest,RedefineAnotherType) { this->redefine_another_type(); }
TYPED_TEST_P(AnyParamTest,Defined) { this->defined(); }
TYPED_TEST_P(AnyParamTest,OriginName) { this->origin_name(); }

REGISTER_TYPED_TEST_CASE_P(AnyParamTest,
                           AccessConst,
                           AccessNonconst,
                           ExplicitCast,
                           Exists,
                           ExistsSame,
                           ImplicitCast,
                           ImplicitAssign,
                           ExplicitCastToLarger,
                           ImplicitCastToLarger,
                           ExistsLarger,
                           ExplicitCastToSmaller,
                           ImplicitCastToSmaller,
                           ExistsSmaller,
                           ExplicitCastToIncompatible,
                           ImplicitCastToIncompatible,
                           ExistsIncompatible,
                           AssignSameType,
                           AssignSmallerType,
                           AssignLargerType,
                           AssignIncompatibleType,
                           RedefineSameType,
                           RedefineAnotherType,
                           Defined,
                           OriginName
    );
