#!/usr/bin/python3
"""A helper program to generate tests for equality/inequality between dictionary values"""

from itertools import \
    combinations_with_replacement as tri_prod, \
    product

domains=(('neg_long',),
         ('neg_int', 'neg_long_is'),
         ('pos_int', 'uint_is', 'pos_long_is', 'ulong_is'),
         ('pos_long', 'ulong_ls'),
         ('ulong',))

# Mapping between domain names and function name parts
dom_name={
    'neg_long' : 'NegLong',
    'neg_int' : 'NegInt',
    'pos_int' : 'PosInt',
    'pos_long' : 'PosLong',
    'ulong' : 'ULong' }

def get_refs(v): return (v, 'p["%s"]'%v)

def get_ref_pairs(lhs,rhs):
    p=product(get_refs(lhs), get_refs(rhs))
    p.__next__()
    return p

for lhs_domain, rhs_domain in tri_prod(domains, 2):
    if lhs_domain is rhs_domain:
        print("// Equalities within domain %s" % lhs_domain[0])
        print("TEST_F(MyTest, eq%s) {" % dom_name[lhs_domain[0]])
        for lhs_val, rhs_val in tri_prod(lhs_domain, 2):
            print("    // Same values:")
            for lhs,rhs in get_ref_pairs(lhs_val,rhs_val):
                print('    EXPECT_TRUE(  %s==%s );' % (lhs,rhs))
                print('    EXPECT_FALSE( %s!=%s );' % (lhs,rhs))

            print("    // Different values:")
            for lhs,rhs in get_ref_pairs(lhs_val,rhs_val+"1"):
                print('    EXPECT_TRUE(  %s!=%s );' % (lhs,rhs))
                print('    EXPECT_FALSE( %s==%s );' % (lhs,rhs))
    else:
        print("// Equalities between domains %s:%s" % (lhs_domain[0],rhs_domain[0]))
        print("TEST_F(MyTest, eq%s%s) {" % (dom_name[lhs_domain[0]], dom_name[rhs_domain[0]]))
        for lhs_val, rhs_val in product(lhs_domain, rhs_domain):
            for lhs, rhs in get_ref_pairs(lhs_val,rhs_val):
                print('    EXPECT_TRUE(  %s!=%s );' % (lhs,rhs))
                print('    EXPECT_FALSE( %s==%s );' % (lhs,rhs))

    # endif
    print("}\n")
# endfor
