from alps_numeric import *


a = value_with_error(1.,0.1)
b = value_with_error(2.,0.2)
c = value_with_error(3.,0.3)

M = numpy.array([1,2,3])
N = numpy.array([1.,2.,3.])

X = vector_with_error()
X.push_back(a)
X.push_back(b)
X.push_back(c)

Y = vector_of_value_with_error()
Y = convert2_vector_of_value_with_error(X)

Z = numpy.array([a,b,c])


print "\n"
print "numpy array M: \n" + str(M) + "\n"
print "numpy array N: \n" + str(N) + "\n"
print "vector_with_error X: \n" + str(X)
print "vector_of_value_with_error Y: \n" + str(Y) + "\n"
print "numpy array Z: \n" + str(Z)


global_test_function = 'print "\\n\\nTEST_OP\\n" \n\
print "------------------------------------------------------------------------------\\n" \n\
print "TEST_OP(1)  =\\t " + str(TEST_OP(1)) \n\
print "TEST_OP(1.) =\\t " + str(TEST_OP(1.)) \n\
print "TEST_OP(value_with_error(1.,0.1)) =\t " + str(TEST_OP(value_with_error(1.,0.1))) + "\\n" \n\
print "TEST_OP(M) =\\t" + str(TEST_OP(M)) + "\\n" \n\
print "TEST_OP(N) =\\t" + str(TEST_OP(N)) + "\\n" \n\
print "TEST_OP(X) =\\n" + str(TEST_OP(X)) \n\
print "TEST_OP(Y) =\\n" + str(TEST_OP(Y)) + "\\n" \n\
print "TEST_OP(Z) =\\t" + str(TEST_OP(Z)) \n\
'

for test_operation in ["sqrt","exp","log","sin","cos","tan","sinh","cosh","tanh"]:
  test_function = global_test_function.replace("TEST_OP",test_operation)
  exec test_function


print '\n### WARNING: numpy arrays do not support "asin", "acos", "atan", "asinh", "acosh", "atanh"'


global_test_function = 'print "\\n\\nTEST_OP\\n" \n\
print "------------------------------------------------------------------------------\\n" \n\
print "TEST_OP(1)  =\\t " + str(TEST_OP(1)) \n\
print "TEST_OP(1.) =\\t " + str(TEST_OP(1.)) \n\
print "TEST_OP(value_with_error(1.,0.1)) =\t " + str(TEST_OP(value_with_error(1.,0.1))) + "\\n" \n\
print "TEST_OP(X) =\\n" + str(TEST_OP(X)) \n\
print "TEST_OP(Y) =\\n" + str(TEST_OP(Y)) + "\\n" \n\
'

for test_operation in ["asin","acos","atan","asinh","acosh"]:
  test_function = global_test_function.replace("TEST_OP",test_operation)
  exec test_function


print '\n### WARNING: "atanh" is having problems in python exportation'


print '\n### WARNING: "sq", "cb", "cbrt" functions are only supported on value_with_array class or its derived classes'


global_test_function = 'print "\\n\\nTEST_OP\\n" \n\
print "------------------------------------------------------------------------------\\n" \n\
print "TEST_OP(value_with_error(1.,0.1)) =\t " + str(TEST_OP(value_with_error(1.,0.1))) + "\\n" \n\
print "TEST_OP(X) =\\n" + str(TEST_OP(X)) \n\
print "TEST_OP(Y) =\\n" + str(TEST_OP(Y)) + "\\n" \n\
'

for test_operation in ["sq","cb","cbrt"]:
  test_function = global_test_function.replace("TEST_OP",test_operation)
  exec test_function




