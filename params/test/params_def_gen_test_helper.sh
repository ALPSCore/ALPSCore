#!/bin/bash

declare -A defstat=([DEF]='with default' [NODEF]='without default')
declare -A dictstat=([N]='not defined in dict' [C]='pre-defined in dict, same type' [W]='pre-defined in dict, different type')
declare -A argstat=([N]='missing' [C]='correct type' [W]='incorrect type')
declare -A redefstat=([N]='not redefined' [C]='redefined with the same type' [W]='redefined with new type')


declare -A defcode
defcode[DEF]='const int deflt_int_val=1111;
    EXPECT_THROW_TRUE_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                            de::type_mismatch);'
defcode[NODEF]='EXPECT_THROW_TRUE_FALSE(par_.define<int>(name, "Int arg without default"),
                            de::type_mismatch);'

declare -A dictcode
dictcode[N]='/* not in dict */'
dictcode[C]='const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;'
dictcode[W]='const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;'

declare -A argcode
argcode[N]='std::string name="no_such_arg";'
argcode[C]='std::string name="an_int";
     const int expected_arg_val=1234;'
argcode[W]='std::string name="simple_string";'

declare -A redefcode
redefcode[N]='/* not redefined */'
redefcode[C]='const int redef_int_value=9999;
    EXPECT_THROW_TRUE_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default"),
de::type_mismatch);'
redefcode[W]='EXPECT_THROW_TRUE_FALSE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
de::type_mismatch);'

idx=0

for def in NODEF DEF; do
    for dict in N C W; do
        for arg in N C W; do 
            for redef in N C W; do

                let idx++
cat <<EOF

/*
   Variant $idx
   the argument is ${argstat[$arg]}
   ${dictstat[$dict]}
   Parameter defined ${defstat[$def]}
   ${redefstat[$redef]}
*/
TEST_F(ParamsTest0, defined${def}dict${dict}arg${arg}redef${redef}) {
    ${argcode[$arg]}

    ${dictcode[$dict]}

    ${defcode[$def]}

    ${redefcode[$redef]}

    ASSERT_TRUE_FALSE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
    EXPECT_EQ(deflt_int_val, actual);
    EXPECT_EQ(expected_arg_val, actual);
}
EOF
            done; done; done; done
