/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file foreach_params.cpp

    @brief Tests the behaviour of apply() and foreach() free functions
*/

#include <alps/params.hpp>
#include "./params_test_support.hpp"

using ::testing::AssertionSuccess;
using ::testing::AssertionFailure;
using ::testing::AssertionResult;


#include <map>
#include <set>

// #include <boost/variant.hpp>
#include <boost/optional.hpp>
#include <boost/any.hpp>
#include <boost/foreach.hpp>

using boost::optional;
using boost::variant;
typedef std::vector<int> intvec;
using std::string;

class ParamsTest0 : public ::testing::Test {
  public:
    arg_holder args_;
    alps::params par_;

    intvec value_for_vec_;
    intvec default_for_vec_;
    
    ParamsTest0() : args_(), par_(), value_for_vec_(), default_for_vec_() {
        args_
            .add("--trigger")
            .add("int_param=1")
            .add("int_with_default=2")
            .add("str_param=string")
            .add("vec=1,2,3,4");

        // FIXME:C++11 will make it more concise
        value_for_vec_.push_back(1);
        value_for_vec_.push_back(2);
        value_for_vec_.push_back(3); 
        value_for_vec_.push_back(4); 

        // FIXME:C++11 will make it more concise
        default_for_vec_.push_back(3);
        default_for_vec_.push_back(2);
        default_for_vec_.push_back(1); 
        
        alps::params p(args_.argc(), args_.argv());
        p
            .define("trigger", "On/off")
            .define<int>("int_param", "An int")
            .define<int>("int_with_default", 99, "Int with def")
            .define<double>("missing_double_with_default", 1.25, "Dbl with def")
            .define<string>("str_param", "A string")
            .define<intvec>("vec", "A vector")
            .define<intvec>("missing_vec_with_default", default_for_vec_, "Vector with deflt")
            ;

        EXPECT_TRUE(p.ok()) << "Something wrong with param object initialization";
        swap(p, par_);
    }

};

struct param_capture {
    string name_;
    boost::any val_;
    boost::any defval_;
    string descr_;

    template <typename T>
    void set(const string& name, const optional<T>& val,
             const optional<T>& defval, const string& descr)
    {
        name_=name;
        if (val) val_=*val;
        if (defval) defval_=*defval;
        descr_=descr;
    }

    template <typename T>
    AssertionResult check_type() const {
        const T* valptr=boost::any_cast<T>(&val_);
        const T* defptr=boost::any_cast<T>(&defval_);
        return (valptr || defptr) ?
            AssertionSuccess() :
            AssertionFailure() << "Type mismatch";
    }
    
    AssertionResult check_name(const string& expected) const {
        return (name_==expected)?
            AssertionSuccess() << "Names match" :
            AssertionFailure() << "Name mismatch: expected \"" << expected
                               << "\" got \"" << name_ << "\"";
    }

    AssertionResult check_descr(const string& expected) const {
        return (descr_==expected)?
            AssertionSuccess() << "Descriptions match" :
            AssertionFailure() << "Description mismatch: expected \"" << expected
                               << "\" got \"" << descr_ << "\"";
    }

    AssertionResult has_val() const {
        return val_.empty() ?
            AssertionFailure() << "Value is empty" :
            AssertionSuccess() << "Value is not empty";
    }
    
    AssertionResult has_defval() const {
        return defval_.empty() ?
            AssertionFailure() << "Default value is empty" :
            AssertionSuccess() << "Default value is not empty";
    }
    
    template <typename ExpectedType>
    AssertionResult check_val(const ExpectedType& expected) const {
        if (!has_val()) return has_val();
        const ExpectedType actual=boost::any_cast<ExpectedType>(val_);
        return (actual==expected)?
            AssertionSuccess() << "Values match" :
            AssertionFailure() << "Values mismatch: expected " << expected
                               << " got " << actual;
    }

    template <typename ExpectedType>
    AssertionResult check_defval(const ExpectedType& expected) const {
        if (!has_defval()) return has_defval();
        const ExpectedType actual=boost::any_cast<ExpectedType>(defval_);
        return (actual==expected)?
            AssertionSuccess() << "Default values match" :
            AssertionFailure() << "Default values mismatch: expected " << expected
                               << " got " << actual;
    }
    
};

template <typename ExpectedType>
struct apply_test_functor {
    param_capture& p_capture_;
    
    apply_test_functor(param_capture& pc): p_capture_(pc) {}

    void operator()(const std::string& name,
                    boost::optional<ExpectedType> const& val,
                    boost::optional<ExpectedType> const& defval,
                    const std::string& descr) const
    {
        p_capture_.set(name, val, defval, descr);
    }

    template <typename T>
    void operator()(const std::string& name,
                    boost::optional<T> const& val,
                    boost::optional<T> const& defval,
                    const std::string& descr) const
    {
        FAIL() << "Wrong type";
    }

};


/* implementation */

template <typename F>
struct caller {
    typedef void result_type;
    const F& fn_;
    const alps::params& par_;
    const std::string& name_;

    caller(const F& f, const alps::params& p, const std::string&  name) : fn_(f), par_(p), name_(name) {}

    template <typename T>
    void operator()(const T& val) const {
        fn_(name_, boost::optional<T>(val), boost::optional<T>(val), par_.get_descr(name_));
    }

};

template <typename F>
void apply(const alps::params& p, const std::string& optname, const F& f)
{
    alps::params::const_iterator it=p.find(optname);
    if (p.end()==it) throw std::runtime_error("Name not found"); // FIXME: be more specific
    
    apply_visitor(caller<F>(f, p, optname), it);
    // f("trigger", optional<bool>(true), optional<bool>(false), "On/off");
}

/* end implementation */


TEST_F(ParamsTest0, applyTrigger) {
    param_capture pc;
    apply(par_, "trigger", apply_test_functor<bool>(pc));

    EXPECT_TRUE(pc.check_type<bool>());
    EXPECT_TRUE(pc.check_name("trigger"));
    EXPECT_TRUE(pc.check_descr("On/off"));
    EXPECT_TRUE(pc.check_val(true));
    EXPECT_TRUE(pc.check_defval(false));
}

TEST_F(ParamsTest0, applyIntParam) {
    param_capture pc;
    apply(par_, "int_param", apply_test_functor<int>(pc));

    EXPECT_TRUE(pc.check_type<int>());
    EXPECT_TRUE(pc.check_name("int_param"));
    EXPECT_TRUE(pc.check_descr("An int"));
    EXPECT_TRUE(pc.check_val(1));
    EXPECT_FALSE(pc.has_defval());
}

TEST_F(ParamsTest0, applyIntWithDefParam) {
    param_capture pc;
    apply(par_, "int_with_default", apply_test_functor<int>(pc));

    EXPECT_TRUE(pc.check_type<int>());
    EXPECT_TRUE(pc.check_name("int_with_default"));
    EXPECT_TRUE(pc.check_descr("Int with def"));
    EXPECT_TRUE(pc.check_val(2));
    EXPECT_TRUE(pc.check_defval(99));
}

TEST_F(ParamsTest0, applyMissingDblWithDefParam) {
    param_capture pc;
    apply(par_, "missing_double_with_default", apply_test_functor<double>(pc));

    EXPECT_TRUE(pc.check_type<double>());
    EXPECT_TRUE(pc.check_name("missing_double_with_default"));
    EXPECT_TRUE(pc.check_descr("Dbl with def"));
    EXPECT_FALSE(pc.has_val());
    EXPECT_TRUE(pc.check_defval(1.25));
}

TEST_F(ParamsTest0, applyStringParam) {
    param_capture pc;
    apply(par_, "str_param", apply_test_functor<string>(pc));

    EXPECT_TRUE(pc.check_type<string>());
    EXPECT_TRUE(pc.check_name("str_param"));
    EXPECT_TRUE(pc.check_descr("A string"));
    EXPECT_TRUE(pc.check_val(string("string")));
    EXPECT_FALSE(pc.has_defval());
}

TEST_F(ParamsTest0, applyVecParam) {
    param_capture pc;
    apply(par_, "vec", apply_test_functor<intvec>(pc));

    EXPECT_TRUE(pc.check_type<intvec>());
    EXPECT_TRUE(pc.check_name("vec"));
    EXPECT_TRUE(pc.check_descr("A vector"));
    EXPECT_TRUE(pc.check_val(value_for_vec_));
    EXPECT_FALSE(pc.has_defval());
}

TEST_F(ParamsTest0, applyVecDefaultParam) {
    param_capture pc;
    apply(par_, "vec", apply_test_functor<intvec>(pc));

    EXPECT_TRUE(pc.check_type<intvec>());
    EXPECT_TRUE(pc.check_name("missing_vec_with_default"));
    EXPECT_TRUE(pc.check_descr("Vector with deflt"));
    EXPECT_FALSE(pc.has_val());
    EXPECT_TRUE(pc.check_defval(default_for_vec_));
}

TEST_F(ParamsTest0, applyNonexistent) {
    param_capture pc;
    EXPECT_ANY_THROW(apply(par_, "no_such_param", apply_test_functor<int>(pc)));
}
