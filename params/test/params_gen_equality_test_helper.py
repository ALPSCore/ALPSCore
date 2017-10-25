#!/usr/bin/python3
"""A helper program to generate tests for equality/inequality between dictionary values"""

from itertools import \
    combinations_with_replacement as tri_prod, \
    product

num_domains=(('neg_long',),
             ('neg_int', 'neg_long_is'),
             ('pos_int', 'uint_is', 'pos_long_is', 'ulong_is'),
             ('pos_uint',),
             ('pos_long', 'ulong_ls'),
             ('pos_ulong',))

incompat_types=('my_bool', 'my_int', 'my_string', 'my_vec', 'my_pair')

# Mapping between domain names and function name parts
dom_name={
    'neg_long' : 'NegLong',
    'neg_int' : 'NegInt',
    'pos_int' : 'PosInt',
    'pos_uint' : 'PosUint',
    'pos_long' : 'PosLong',
    'pos_ulong' : 'ULong',
    'my_int' : 'Int',
    'my_string' : 'String',
    'my_bool'  : 'Bool',
    'my_vec' : 'Vec',
    'my_pair' : 'Pair',
    'my_float' : 'Float',
    'my_double' : 'Double'}

labels=('Left','Right','Both')

def generate_num_equalities():
    vars2refs={'Left' :lambda lhs,rhs: ('cdict_["%s"]'%lhs, "+"+rhs),
               'Right':lambda lhs,rhs: ("+"+lhs, 'cdict_["%s"]'%rhs),
               'Both' : lambda lhs,rhs: ('cdict_["%s"]'%lhs, 'cdict_["%s"]'%rhs)}
    
    for lhs_domain, rhs_domain in tri_prod(num_domains, 2):
        if lhs_domain is rhs_domain:
            print("// Equalities within domain %s" % lhs_domain[0])
            # print("TEST_F(MyTest, eq%s) {" % dom_name[lhs_domain[0]])
            
            for lab in labels:
                v2r=vars2refs[lab]
                print("TEST_F(MyTest, eq%s%s) {" % (dom_name[lhs_domain[0]], lab))
                for lhs, rhs in tri_prod(lhs_domain, 2):
                    print("    // Same values:")
                    print('    EXPECT_TRUE(  %s==%s );' % v2r(lhs,rhs))
                    print('    EXPECT_FALSE( %s!=%s );' % v2r(lhs,rhs))
    
                    print("    // Different values:")
                    print('    EXPECT_TRUE(  %s!=%s );' % v2r(lhs,rhs+'1'))
                    print('    EXPECT_FALSE( %s==%s );' % v2r(lhs,rhs+'1'))
                print("}\n")
        else:
            print("// Equalities between domains %s:%s" % (lhs_domain[0],rhs_domain[0]))
            #print("TEST_F(MyTest, eq%s%s) {" % (dom_name[lhs_domain[0]], dom_name[rhs_domain[0]]))
            for lab in labels:
                v2r=vars2refs[lab]
                print("TEST_F(MyTest, eq%s%s%s) {" % (dom_name[lhs_domain[0]], dom_name[rhs_domain[0]], lab))
                for lhs, rhs in product(lhs_domain, rhs_domain):
                    print('    EXPECT_TRUE(  %s!=%s );' % v2r(lhs,rhs))
                    print('    EXPECT_FALSE( %s==%s );' % v2r(lhs,rhs))
                print("}\n")
    return

def generate_obj_equalities():
    vars2refs={'Left' :lambda lhs,rhs: ('cdict_["%s"]'%lhs, rhs),
               'Right':lambda lhs,rhs: (lhs, 'cdict_["%s"]'%rhs),
               'Both' : lambda lhs,rhs: ('cdict_["%s"]'%lhs, 'cdict_["%s"]'%rhs)}

    for lhs, rhs in tri_prod(incompat_types, 2):
        if lhs is rhs:
            print("// Equalities within same type %s" % lhs)
            for lab in labels:
                v2r=vars2refs[lab]
                print("TEST_F(MyTest, eq%s%s) {" % (dom_name[lhs],lab))
                print("    // Same values:")
                print('    EXPECT_TRUE(  %s==%s );' % v2r(lhs,rhs))
                print('    EXPECT_FALSE( %s!=%s );' % v2r(lhs,rhs))
    
                print("    // Different values:")
                print('    EXPECT_TRUE(  %s!=%s );' % v2r(lhs,rhs+'1'))
                print('    EXPECT_FALSE( %s==%s );' % v2r(lhs,rhs+'1'))
                print("}\n")
        else:
            print("// Equalities between different types %s:%s" % (lhs,rhs))
            for lab in labels:
                v2r=vars2refs[lab]
                print("TEST_F(MyTest, eq%s%s%s) {" % (dom_name[lhs],dom_name[rhs],lab))
                print('    EXPECT_THROW(  %s!=%s, de::type_mismatch );' % v2r(lhs,rhs))
                print('    EXPECT_THROW(  %s==%s, de::type_mismatch );' % v2r(lhs,rhs))
                print("}\n")

generate_num_equalities()
generate_obj_equalities()
