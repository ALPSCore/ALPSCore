from pyalea import *

print "\nTesting value_with_error"
print "\n------------------------\n"

a = value_with_error(0.81,0.1)
b = value_with_error(1.21,0.15)
c = value_with_error(-1.5,0.2)

print "Initialization:\n"
print "a:\t" + str(a)
print "b:\t" + str(b)
print "c:\t" + str(c)

print "\n"

print "Operation:\n"

a += b
print "a += b:\t" + str(a)
a = value_with_error(1.2,0.1)

a -= b
print "a -= b:\t" + str(a)
a = value_with_error(1.2,0.1)

a *= b
print "a *= b:\t" + str(a)
a = value_with_error(1.2,0.1)

a /= b
print "a /= b:\t" + str(a)
a = value_with_error(1.2,0.1)

print "\n"

a += 2.
print "a += 2.:\t" + str(a)
a = value_with_error(1.2,0.1)

a -= 2.
print "a -= 2.:\t" + str(a)
a = value_with_error(1.2,0.1)

a *= 2.
print "a *= 2.:\t" + str(a)
a = value_with_error(1.2,0.1)

a /= 2.
print "a /= 2.:\t" + str(a)
a = value_with_error(1.2,0.1)

print "\n"

print "a + b:\t" + str(a / b)
print "a + 2.:\t" + str(a / 2.)
print "2. + a:\t" + str(2. / a)
print "a - b:\t" + str(a / b)
print "a - 2.:\t" + str(a / 2.)
print "2. - a:\t" + str(2. / a)
print "a * b:\t" + str(a / b)
print "a * 2.:\t" + str(a / 2.)
print "2. * a:\t" + str(2. / a)
print "a / b:\t" + str(a / b)
print "a / 2.:\t" + str(a / 2.)
print "2. / a:\t" + str(2. / a)

print "\n"

print "-a:\t" + str(-a)
print "abs(c):\t" + str(abs(c))

print "\n"

print "pow(a,2.71):\t" + str(pow(a,2.71))
print "a.sq()\t" + str(a.sq())
print "a.sqrt()\t" + str(a.sqrt())
print "a.cb()\t" + str(a.cb())
print "a.cbrt()\t" + str(a.cbrt())
print "a.exp()\t" + str(a.exp())
print "a.log()\t" + str(a.log())

print "a.sin()\t" + str(a.sin())
print "a.cos()\t" + str(a.cos())
print "a.tan()\t" + str(a.tan())
print "a.asin()\t" + str(a.asin())
print "a.acos()\t" + str(a.acos())
print "a.atan()\t" + str(a.atan())
print "a.sinh()\t" + str(a.sinh())
print "a.cosh()\t" + str(a.cosh())
print "a.tanh()\t" + str(a.tanh())
print "a.asinh()\t" + str(a.asinh())
print "b.acosh()\t" + str(b.acosh())
print '\n### WARNING : "atanh" is having problems in python exportation.'   
#print "a.atanh()\t" + repr(a.atanh())    

print "\n"


print "\nTesting vector_with_error"
print "\n------------------------\n"

print "Manipulation\n"

M = vector()
M.append(2.3)
M.append(1.2)
M.append(0.7)

E = vector()
E.append(0.01)
E.append(0.02)
E.append(0.07)

X = vector_with_error(M,E)

print "X:\n" + str(X)

X.push_back(a)
print "Process 1: X.push_back(a):\n" + str(X)

X.pop_back()
print "Process 2: X.pop_back():\n" + str(X)

print "Process 2: X.at(0):\t" + str(X.at(0))

X.insert(0,b)
print "Process 3: X.insert(0,b):\n" + str(X)

X.erase(1)
print "Process 4: X.erase(1):\n" + str(X)

X.clear()
print "Process 5: X.clear():\n" + str(X)


print "Operation\n"

X = vector_with_error(M,E)
Y = X+1.

print "X:\n" + str(X)
print "Y:\n" + str(Y)
print "E:\n" + str(E)

print "X + Y:\n" + str(X+Y)
print "X + E:\n" + str(X+E)
print "E + X:\n" + str(E+X)
print "X + 2.:\n" + str(X+2.)
print "2. + X:\n" + str(2.+X)

print "X + Y:\n" + str(X+Y)
print "X + E:\n" + str(X+E)
print "E + X:\n" + str(E+X)
print "X + 2.:\n" + str(X+2.)
print "2. + X:\n" + str(2.+X)

print "X / Y:\n" + str(X/Y)
print "X / E:\n" + str(X/E)
print "E / X:\n" + str(E/X)
print "X / 2.:\n" + str(X/2.)
print "2. / X:\n" + str(2./X)

print "X / Y:\n" + str(X/Y)
print "X / E:\n" + str(X/E)
print "E / X:\n" + str(E/X)
print "X / 2.:\n" + str(X/2.)
print "2. / X:\n" + str(2./X)

print "-X:\n" + str(-X)
print "abs(X):\n" + str(X)

print "pow(X,2.71):\n" + str(pow(X,2.71))
print "X.sq():\n" + str(X.sq())
print "X.sqrt():\n" + str(X.sqrt())
print "X.cb():\n" + str(X.cb())
print "X.cbrt():\n" + str(X.cbrt())
print "X.exp():\n" + str(X.exp())
print "X.log():\n" + str(X.log())

print "X.sin():\n" + str(X.sin())
print "X.cos():\n" + str(X.cos())
print "X.tan():\n" + str(X.tan())
print "X.asin():\n" + str(X.asin())
print "X.acos():\n" + str(X.acos())
print "X.atan():\n" + str(X.atan())
print "X.sinh():\n" + str(X.sinh())
print "X.cosh():\n" + str(X.cosh())
print "X.tanh():\n" + str(X.tanh())
print "X.asinh():\n" + str(X.asinh())
print "Y.acosh():\n" + str(Y.acosh())
print "NOTE: atanh IS NOT OKAY IN PYTHON EXPORTATION!!!"
#print "X.atanh():\n" + str(X.atanh())

print "\n"

print "\nvector_of_value_with_error"
print "\n--------------------------\n"

print "Manipulation:\n"

G = vector_of_value_with_error()
G.append(a)
G.append(b)
G.append(c)

print "G:\n" + str(G) + "\n"

H = G[1:3]
print "H = G[1:3]:\n" + str(H) + "\n"

del H[1]
print "del H[1], H:\n" + str(H) + "\n"

H.append(a)
print "H.append(a):\n" + str(H) + "\n"

H.extend(G)
print "H.extend(G):\n" + str(H) + "\n"

del H[3:]
print "del H[3:], H:\n" + str(H) + "\n"

print "Operations:\n"

G = convert2_vector_of_value_with_error(X)
H = convert2_vector_of_value_with_error(Y)

print "G:\n" + str(G) + "\n"
print "H:\n" + str(H) + "\n"
print "E:\n" + str(E) + "\n"

print "G + H:\n" + str(G+H) + "\n"
print "G + E:\n" + str(G+E) + "\n"
print "E + G:\n" + str(E+G) + "\n"
print "G + 2.:\n" + str(G+2.) + "\n"
print "2. + G:\n" + str(2.+G) + "\n"

print "G + H:\n" + str(G+H) + "\n"
print "G + E:\n" + str(G+E) + "\n"
print "E + G:\n" + str(E+G) + "\n"
print "G + 2.:\n" + str(G+2.) + "\n"
print "2. + G:\n" + str(2.+G) + "\n"

print "G / H:\n" + str(G/H) + "\n"
print "G / E:\n" + str(G/E) + "\n"
print "E / G:\n" + str(E/G) + "\n"
print "G / 2.:\n" + str(G/2.) + "\n"
print "2. / G:\n" + str(2./G) + "\n"

print "G / H:\n" + str(G/H) + "\n"
print "G / E:\n" + str(G/E) + "\n"
print "E / G:\n" + str(E/G) + "\n"
print "G / 2.:\n" + str(G/2.) + "\n"
print "2. / G:\n" + str(2./G) + "\n"

print "-G:\n" + str(-G) + "\n"
print "abs(G):\n" + str(G) + "\n"

print "pow(G,2.71):\n" + str(pow(G,2.71)) + "\n"
print "G.sq():\n" + str(G.sq()) + "\n"
print "G.sqrt():\n" + str(G.sqrt()) + "\n"
print "G.cb():\n" + str(G.cb()) + "\n"
print "G.cbrt():\n" + str(G.cbrt()) + "\n"
print "G.exp():\n" + str(G.exp()) + "\n"
print "G.log():\n" + str(G.log()) + "\n"

print "G.sin():\n" + str(G.sin()) + "\n"
print "G.cos():\n" + str(G.cos()) + "\n"
print "G.tan():\n" + str(G.tan()) + "\n"
print "G.asin():\n" + str(G.asin()) + "\n"
print "G.acos():\n" + str(G.acos()) + "\n"
print "G.atan():\n" + str(G.atan()) + "\n"
print "G.sinh():\n" + str(G.sinh()) + "\n"
print "G.cosh():\n" + str(G.cosh()) + "\n"
print "G.tanh():\n" + str(G.tanh()) + "\n"
print "G.asinh():\n" + str(G.asinh()) + "\n"
print "H.acosh():\n" + str(H.acosh()) + "\n"
print "NOTE: atanh IS NOT OKAY IN PYTHON EXPORTATION!!!"
#print "G.atanh():\n" + str(G.atanh()) + "\n"

print "\n"

print "\nSwitching from vector_with_error to vector_of_value_with_error and vice versa"
print "\n-----------------------------------------------------------------------------"

V1 = vector_with_error()
V1 = convert2_vector_with_error(G)
print "V1 = vector_with_error()\nV1 = convert2_vector_with_error(G)\n\nV1:\n" + str(V1)

V2 = vector_of_value_with_error()
V2 = convert2_vector_of_value_with_error(X)
print "V2 = vector_of_value_with_error()\nV2 = convert2_vector_with_error(X)\n\nV2:\n" + str(V2)








