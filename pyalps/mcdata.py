# ****************************************************************************
# 
# ALPS Project: Algorithms and Libraries for Physics Simulations
# 
# ALPS Libraries
# 
# Copyright (C) 2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch> ,
#                       Matthias Troyer <troyer@itp.phys.ethz.ch>
#
# This software is part of the ALPS libraries, published under the ALPS
# Library License; you can use, redistribute it and/or modify it under
# the terms of the license, either version 1 or (at your option) any later
# version.
#  
# You should have received a copy of the ALPS Library License along with
# the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
# available from http://alps.comp-phys.org/.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
# SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
# FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# ****************************************************************************

from pyalps.pyalea import *
import numpy as np 

print "\nTesting MCScalarData"
print "\n------------------------\n"

a = MCScalarData(0.81,0.1)
b = MCScalarData(1.21,0.15)
c = MCScalarData(-1.5,0.2)

print "Initialization:\n"
print "a:\t" + str(a)
print "b:\t" + str(b)
print "c:\t" + str(c)

print "\n"

print "Operation:\n"

a += b
print "a += b:\t" + str(a)
a = MCScalarData(1.2,0.1)

a -= b
print "a -= b:\t" + str(a)
a = MCScalarData(1.2,0.1)

a *= b
print "a *= b:\t" + str(a)
a = MCScalarData(1.2,0.1)

a /= b
print "a /= b:\t" + str(a)
a = MCScalarData(1.2,0.1)

print "\n"

a += 2.
print "a += 2.:\t" + str(a)
a = MCScalarData(1.2,0.1)

a -= 2.
print "a -= 2.:\t" + str(a)
a = MCScalarData(1.2,0.1)

a *= 2.
print "a *= 2.:\t" + str(a)
a = MCScalarData(1.2,0.1)

a /= 2.
print "a /= 2.:\t" + str(a)
a = MCScalarData(1.2,0.1)

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
# print "a.atanh()\t" + repr(a.atanh())

print "\n"
print "\nTesting MCVectorData"
print "\n------------------------\n"

print "Manipulation\n"

X = MCVectorData(np.array([2.3, 1.2, 0.7]), np.array([0.01, 0.01, 0.01]))
Y = X+1.

print "X:\n" + str(X)
print "Y:\n" + str(Y)

print "X + Y:\n" + str(X+Y)
print "X + 2.:\n" + str(X+2.)
print "2. + X:\n" + str(2.+X)

print "X + Y:\n" + str(X+Y)
print "X + 2.:\n" + str(X+2.)
print "2. + X:\n" + str(2.+X)

print "X / Y:\n" + str(X/Y)
print "X / 2.:\n" + str(X/2.)
print "2. / X:\n" + str(2./X)

print "X / Y:\n" + str(X/Y)
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
# print "X.atanh():\n" + str(X.atanh())

