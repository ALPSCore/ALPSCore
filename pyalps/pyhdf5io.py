 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 #                                                                                 #
 # ALPS Project: Algorithms and Libraries for Physics Simulations                  #
 #                                                                                 #
 # ALPS Libraries                                                                  #
 #                                                                                 #
 # Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   #
 #                                                                                 #
 # This software is part of the ALPS libraries, published under the ALPS           #
 # Library License; you can use, redistribute it and/or modify it under            #
 # the terms of the license, either version 1 or (at your option) any later        #
 # version.                                                                        #
 #                                                                                 #
 # You should have received a copy of the ALPS Library License along with          #
 # the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       #
 # available from http://alps.comp-phys.org/.                                      #
 #                                                                                 #
 #  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 # FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       #
 # SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       #
 # FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     #
 # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     #
 # DEALINGS IN THE SOFTWARE.                                                       #
 #                                                                                 #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import pyalps.hdf5 as hdf5

ar = hdf5.archive('pyngs.h5', 'w')
a = np.array([1, 2, 3]);
b = np.array([1.1, 2.0, 3.5]);
c = np.array([1.1 + 1j, 2.0j, 3.5]);
d = {"a": a, 2 + 3j: "foo"}

ar["/list"] = [1, 2, 3]
ar["/list2"] = [[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]
ar["/dict"] = {"scalar": 1, "numpy": a, "numpycpx": c, "list": [1, 2, 3], "string": "str", 1: 1, 4: d}
ar["/numpy"] = a
ar["/numpy2"] = b
ar["/numpy3"] = c
ar["/numpyel"] = a[0]
ar["/numpyel2"] = b[0]
ar["/numpyel3"] = c[0]
ar["/int"] = int(1)
ar["/long"] = long(1)
ar["/double"] = float(1)
ar["/complex"] = complex(1, 1)
ar["/string"] = "str"
ar["/stringlist"] = ['a','list','of','strings']
ar["/inhomogenious"] = [[1, 2, 3], a, "gurke", [[a, 2, 3], ["x", complex(1, 1)]]]
ar["/inhomogenious2"] = [[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3]]]
ar["/inhomogenious3"] = [np.arange(3), np.arange(5)]
ar["/inhomogenious4"] = [np.arange(3), 10 * np.arange(3)]
ar["/inhomogenious5"] = [range(3), range(5), range(3)]
ar["/numpylist1"] = [np.arange(5), np.arange(5,10)]
ar["/numpylist2"] = [np.arange(5), np.arange(10)]

del ar

ar = hdf5.archive('pyngs.h5', 'r')

childs = ar.list_children('/')
l1 = ar["/list"]
l2 = ar["/list2"]
d1 = ar["/dict"]
n1 = ar["/numpy"]
n2 = ar["/numpy2"]
n3 = ar["/numpy3"]
e1 = ar["/numpyel"]
e2 = ar["/numpyel2"]
e3 = ar["/numpyel3"]
s1 = ar["/int"]
s2 = ar["/long"]
s3 = ar["/double"]
s4 = ar["/complex"]
s5 = ar["/string"]
ls = ar["/stringlist"]
i1 = ar["/inhomogenious"]
i2 = ar["/inhomogenious2"]
i3 = ar["/inhomogenious2"]
i4 = ar["/inhomogenious2"]
i5 = ar["/inhomogenious2"]
nl1 = ar["/numpylist1"]
nl2 = ar["/numpylist2"]

print len(childs)
print repr(l1)
print repr(l2)
print repr(d1)
print repr(n1)
print repr(n2)
print repr(n3)
print repr(e1)
print repr(e2)
print repr(e3)
print repr(ls)
print repr(s1)
print repr(s2)
print repr(s3)
print repr(s4)
print repr(s5)
print repr(i1)
print repr(i2)
print repr(i3)
print repr(i4)
print repr(i5)
print repr(nl1)
print repr(nl2)

del ar
