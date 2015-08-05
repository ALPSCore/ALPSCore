/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Generate a parameter object in a various ways, test conformance to specifications */

#include <boost/lexical_cast.hpp>

// #include "param_generators.hpp"
#include "alps/params.hpp"
#include "gtest/gtest.h"

#include "alps/utilities/temporary_filename.hpp"
#include <alps/hdf5/archive.hpp>



/*
  Trying to generate a parameter object with valid values in a different way.
  That is, p["name"] should have a value X of type T; We can set X=get_value<T>().
  How can we generate a valid value?
  1) Copy of an object with a valid value.
  2) Load of an object with a valid value.
  3) Default construct, or construct from a command line, or file and command line, then explicitly assign.
  That is, the origin can be: default-constructed, command line constructed, file+command line constructed.
  4) Construct, then define with a default value.
  5) Construct from command line and/or file containing the value.

  Whichever way it is constructed, it should pass a series of tests. 

  Generally, a lifecycle of a parameter object could be:
  1) Construct an object (mandatory): default, from a file and/or commandline, by copying, by loading.
  2) Assign a value (optional).
  3) Define with or without default (optional).
  4) Assign a value again (optional) ?
 */

/// Tag for the special case: trigger parameter
struct trigger_tag {};


/// Helper class for data traits: maps non-void type T to {value=true, type=T} and void to {value=false, type=U}.
/// General case
template <typename T, typename U>
struct has_type_helper {
    static const bool value = true;
    typedef T type;
};

/// Helper class for data traits: maps non-void type T to {value=true, type=T} and void to {value=false, type=U}.
/// T:void specialization
template <typename U>
struct has_type_helper<void,U> {
    static const bool value = false;
    typedef U type;
};

/// Base class for data traits.
/** T is the data type; SMALLER_T is a "smaller" data type or void if none; LARGER_T is a "larger" data type or void if none;
    WRONG_T is an incompatible type.
    
    Defines has_larger_type and has_smaller_type constants; incompatible_type, smaller_type and larger_type types.

    If there is no larger/smaller type, defines it as T to keep compiler happy
    (so that a valid code can still be generated, although it should never be executed,
    protected by `if (has_larger_type) {...}` at runtime).

    An alternative solution to avoid generating the code for the missing type altogether is too cumbersome.
*/
template <typename T, typename SMALLER_T, typename LARGER_T, typename WRONG_T>
struct data_trait_base {
    static T get(bool choice) { return T(choice?('A'+.25):('B'+.75)); } // good for T=int,char,double
    typedef T return_type; // good for anything except trigger_tag

    typedef WRONG_T incompatible_type;

    static const bool has_smaller_type=has_type_helper<SMALLER_T,T>::value;
    typedef typename has_type_helper<SMALLER_T,T>::type smaller_type;

    static const bool has_larger_type=has_type_helper<LARGER_T,T>::value;
    typedef typename has_type_helper<LARGER_T,T>::type larger_type;
};

/// Trait: properties of a data type for the purposes of alps::params
template <typename> struct data_trait;

template <>
struct data_trait<char> : public data_trait_base<char, void, int, std::string> {};
    
template <>
struct data_trait<bool> : public data_trait_base<bool, void, int, std::string> {
    static bool get(bool choice) { return choice; } 
};

template <>
struct data_trait<int> : public data_trait_base<int, char, long, std::string> {};

template <>
struct data_trait<long> : public data_trait_base<long, char, double, std::string> {};

template <>
struct data_trait<double> : public data_trait_base<double, int, void, std::string> {};

template <>
struct data_trait<trigger_tag> : public data_trait_base<bool, void, void, int> {};

template <>
struct data_trait<std::string> : public data_trait_base<std::string, void, void, int> {
    static std::string get(bool choice) {return choice?"aaa":"bbb"; }; 
};

template <typename T>
struct data_trait< std::vector<T> > {
  static T get(bool choice) { return std::vector<T>(3, data_trait<T>::get()); } 
    typedef std::vector<T> return_type; 

    static const bool has_larger_type=false;
    typedef return_type larger_type;

    static const bool has_smaller_type=false;
    typedef return_type smaller_type;

    typedef int incompatible_type;
};


/// Format a value of type T as an input string
template <typename T>
std::string input_string(const T& val)
{
    return boost::lexical_cast<std::string>(val); // good for numbers and bool
}

/// Format a string as an input string
std::string input_string(const std::string& val)
{
    return "\""+val+"\""; 
}

/// Format a vector as an input string
template <typename T>
std::string input_string(const std::vector<T>& vec)
{
    typename std::vector<T>::const_iterator it=vec.begin(), end=vec.end();
    std::string out;
    while (it!=end) {
        out += input_string(*it);
        ++it;
        if (it!=end)  out+=",";
    }
    return out;
}


namespace value_choice {
    enum value_choice_type { CONSTRUCTION, DEFINITION, UNDEFINED };
};

/// Parameter object generator; accepts parameter type T, construction policy, definition policy, and which value to take as "correct".
/** A value that a parameter acquires when alps::params object is constructed depends on the construction policy.
    Likewise, it may (or may not) acquire a value as a result of definition policy.
    Therefore, we need a parameter generator template argument that tells which value to take as the "correct" one.
    Contract:
    alps::params* ConstructPolicy<T>::new_param() is expected to return a pointer to new-allocated alps::params object, associating a value choice "true", if any.
    void DefinePolicy<T>::define_param(alps::params&) is expected to somehow define a parameter of type T in the alps::params object, associating a value choice "false", if any.
*/
template <typename T, template<typename> class ConstructPolicy, template<typename> class DefinePolicy, value_choice::value_choice_type WHICH>
class ParamGenerator {
    typedef ConstructPolicy<T> construct_policy_type;
    typedef DefinePolicy<T> define_policy_type;
    public:
    alps::params& param;
    typedef data_trait<T> data_trait_type; 
    typedef typename data_trait_type::return_type value_type;
    value_type expected_value;
    static const value_choice::value_choice_type choice=WHICH;

    ParamGenerator(): param(*construct_policy_type::new_param())
    {
        define_policy_type::define_param(param);
        switch (WHICH) {
        case value_choice::CONSTRUCTION:
            expected_value=data_trait_type::get(true);
            break;
        case value_choice::DEFINITION:
            expected_value=data_trait_type::get(false);
            break;
        default:
            /* do nothing */
            break;
        }
    }

    ~ParamGenerator() { delete &param; } // horrible, but should work most of the time.
};

/// Construction policy: construct the alps::params object using default constructor
template <typename T>
struct DefaultConstruct {
    static alps::params* new_param() { return new alps::params(); }
};

/// Construction policy: construct the alps::params object from a commandline containing the value of the parameter
template <typename T>
struct PresentCmdlineConstruct {
    static alps::params* new_param() {
        std::string arg="--param=" + input_string(data_trait<T>::get(true));
        const char* argv[]={"progname", arg.c_str()};
        const int argc=sizeof(argv)/sizeof(*argv);
        return new alps::params(argc, argv);
    }
};

/// Construction policy: construct the alps::params object from a commandline with a trigger parameter
template <>
struct PresentCmdlineConstruct<trigger_tag> {
    static alps::params* new_param() {
        const char* argv[]={"progname", "--param"};
        const int argc=sizeof(argv)/sizeof(*argv);
        return new alps::params(argc, argv);
    }
};

/// Construction policy: construct the alps::params object from a commandline NOT containing the value of the parameter
template <typename T>
struct MissingCmdlineConstruct {
    static alps::params* new_param() {
        const char* argv[]={"progname", "--otherparam=0"};
        const int argc=sizeof(argv)/sizeof(*argv);
        return new alps::params(argc, argv);
    }
};

/// Definition policy: define the parameter by direct assignment
template <typename T>
struct AssignDefine {
    static void define_param(alps::params& p) {
        p["param"]=data_trait<T>::get(false);
    }
};

/// Definition policy: define the parameter with a default value
template <typename T>
struct DefaultDefine {
    static void define_param(alps::params& p) {
        p.define<T>("param", data_trait<T>::get(false), "A parameter with a default");
    }
};

/// Definition policy: define the parameter without a default value
template <typename T>
struct NoDefaultDefine {
    static void define_param(alps::params& p) {
        p.define<T>("param", "A parameter without a default");
    }
};

/// Definition policy: define the trigger parameter (without a default value)
template <>
struct NoDefaultDefine<trigger_tag> {
    static void define_param(alps::params& p) {
        p.define("param", "A trigger parameter");
    }
};

/// Definition policy: do not define a parameter at all
template <typename T>
struct NoDefine {
    static void define_param(alps::params& p) { }
};




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
TYPED_TEST_P(ValidParamTest,ImplicitCast ) { this->implicit_cast(); }
TYPED_TEST_P(ValidParamTest,ExplicitCastToLarger ) { this->explicit_cast_to_larger(); }
TYPED_TEST_P(ValidParamTest,ImplicitCastToLarger ) { this->implicit_cast_to_larger(); }
TYPED_TEST_P(ValidParamTest,ExplicitCastToSmaller ) { this->explicit_cast_to_smaller(); }
TYPED_TEST_P(ValidParamTest,ImplicitCastToSmaller ) { this->implicit_cast_to_smaller(); }
TYPED_TEST_P(ValidParamTest,ExplicitCastToIncompatible ) { this->explicit_cast_to_incompatible(); }
TYPED_TEST_P(ValidParamTest,ImplicitCastToIncompatible ) { this->implicit_cast_to_incompatible(); }
TYPED_TEST_P(ValidParamTest,AccessNonexistentConst ) { this->access_nonexistent_const(); }
TYPED_TEST_P(ValidParamTest,AccessNonexistent ) { this->access_nonexistent(); }
TYPED_TEST_P(ValidParamTest,AssignNonexistent ) { this->assign_nonexistent(); }
TYPED_TEST_P(ValidParamTest,AssignSameType ) { this->assign_same_type(); }
TYPED_TEST_P(ValidParamTest,AssignSmallerType ) { this->assign_smaller_type(); }
TYPED_TEST_P(ValidParamTest,AssignLargerType ) { this->assign_larger_type(); }
TYPED_TEST_P(ValidParamTest,AssignIncompatibleType ) { this->assign_incompatible_type(); }
TYPED_TEST_P(ValidParamTest,RedefineSameType ) { this->redefine_same_type(); }
TYPED_TEST_P(ValidParamTest,RedefineAnotherType ) { this->redefine_another_type(); }
TYPED_TEST_P(ValidParamTest,SaveRestore ) { this->save_restore(); }

REGISTER_TYPED_TEST_CASE_P(ValidParamTest,
                           ExplicitCast,
                           ImplicitCast,
                           ExplicitCastToLarger,
                           ImplicitCastToLarger,
                           ExplicitCastToSmaller,
                           ImplicitCastToSmaller,
                           ExplicitCastToIncompatible,
                           ImplicitCastToIncompatible,
                           AccessNonexistentConst,
                           AccessNonexistent,
                           AssignNonexistent,
                           AssignSameType,
                           AssignSmallerType,
                           AssignLargerType,
                           AssignIncompatibleType,
                           RedefineSameType,
                           RedefineAnotherType,
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

