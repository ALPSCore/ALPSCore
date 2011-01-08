# ****************************************************************************
# 
# ALPS Project: Algorithms and Libraries for Physics Simulations
# 
# ALPS Libraries
# 
# Copyright (C) 2010 by Lukas Gamper <gamperl@gmail.com>
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


import pyalps.hdf5 as h5
import numpy as np
import sys

def write():
    ar = h5.oArchive("test.h5")
    ar.write("/int", 9)
    ar.write("/double", 9.123)
    ar.write("/cplx", complex(1, 2))
    ar.write("/str", "test")
    ar.write("/np/int", np.array([1, 2, 3]))
    ar.write("/np2/int", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

def read():
    ar = h5.iArchive("test.h5")
    i = ar.read("/int")
    d = ar.read("/double")
    c = ar.read("/cplx")
    s = ar.read("/str")
    n = ar.read("/np/int")
    if ar.extent("/int")[0] != 1 or ar.extent("/cplx")[0] != 1 or  ar.extent("/np/int")[0] != 3:
        raise Exception('invalid extent')
    if type(i) != int or type(d) != float or type(c) != complex or type(s) != str:
        raise Exception('invalid type')
    if i != 9 or d - 9.123 > 0.001 or s != "test" or np.any(n != np.array([1, 2, 3])):
        raise Exception('invalid value')

try:
    write();
    read();
    print "SUCCESS"
except Exception, e:
    print "ERROR"
    sys.exit(e)
