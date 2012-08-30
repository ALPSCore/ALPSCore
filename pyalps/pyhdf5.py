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

def write(ar):
    ar.write("/int", 9)
    ar.write("/double", 9.123)
    ar.write("/cplx", complex(1, 2))
    ar.write("/str", "test")
    ar.write("/np/int", np.array([1, 2, 3]))
    ar.write("/np2/int", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    ar.write("/np/cplx", np.array([[1 + 1j,2 +2j ],[3 + 3j,4 + 4j]]))
    
    ar.create_group("/my/group")
    ar.write("/my/double", 9.123)
    
    ar.delete_group("/my/group")
    ar.delete_data("/my/double")

def read(ar):
    childs = ar.list_children('/')
    i = ar.read("/int")
    d = ar.read("/double")
    c = ar.read("/cplx")
    s = ar.read("/str")
    n = ar.read("/np/int")
    x = ar.read("/np/cplx")
    
    if len(childs) != 7:
        raise Exception('invalid length of \'/\'')
    if len(ar.extent("/int")) != 1 or ar.extent("/int")[0] != 1 or len(ar.extent("/cplx")) != 1 or ar.extent("/cplx")[0] != 1:
        raise Exception('invalid scalar extent')
    if len(ar.extent("/np/int")) != 1 or ar.extent("/cplx")[0] != 1 or len(ar.extent("/np/cplx")) != 2 or ar.extent("/np/cplx")[0] != 2 or ar.extent("/np/cplx")[1] != 2:
        raise Exception('invalid array extent')
    if type(i) != int or type(d) != float or type(c) != complex or type(s) != str:
        raise Exception('invalid type')
    if i != 9 or d - 9.123 > 0.001 or s != "test" or np.any(n != np.array([1, 2, 3])):
        raise Exception('invalid scalar value')
        
    if np.any(x[0] != np.array([1 + 1j,2 +2j])) or np.any(x[1] != np.array([3 + 3j,4 + 4j])):
        raise Exception('invalid array value')

try:
    oar = h5.oArchive("py.h5")
    write(oar)
    del oar
    
    iar = h5.iArchive("py.h5")
    if iar.is_complex("/int") or not iar.is_complex("/cplx") or not iar.extent("/np/cplx"):
        raise Exception('invalid complex detection')
    read(iar)
    del iar
    
    ar = h5.oArchive("py.h5")
    write(ar)
    read(ar)
    del ar
    
    print "SUCCESS"
except Exception, e:
    print "ERROR"
    sys.exit(e)
