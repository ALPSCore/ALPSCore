# Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
# All rights reserved. Use is subject to license terms. See LICENSE.TXT
# For use in publications, see ACKNOWLEDGE.TXT

import pyalps.hdf5 as h5
import numpy as np
import sys

def write(ar):
    ar["/int"] =  9
    ar["/double"] =  9.123
    ar["/cplx"] =  complex(1, 2)
    ar["/str"] =  "test"
    ar["/np/int"] =  np.array([1, 2, 3])
    ar["/np2/int"] =  np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ar["/np/cplx"] =  np.array([[1 + 1j,2 +2j ],[3 + 3j,4 + 4j]])
    
    ar.create_group("/my/group")
    ar["/my/double"] = 9.123
    
    ar.delete_group("/my/group")
    ar.delete_data("/my/double")

def read(ar):
    childs = ar.list_children('/')
    i = ar["/int"]
    d = ar["/double"]
    c = ar["/cplx"]
    s = ar["/str"]
    n = ar["/np/int"]
    x = ar["/np/cplx"]
    
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
    oar = h5.archive("py.h5", 'w')
    write(oar)
    del oar
    
    iar = h5.archive("py.h5", 'r')
    if iar.is_complex("/int") or not iar.is_complex("/cplx") or not iar.extent("/np/cplx"):
        raise Exception('invalid complex detection')
    read(iar)
    del iar
    
    ar = h5.archive("py.h5", 'w')
    write(ar)
    read(ar)
    del ar
    
    print "SUCCESS"
except Exception, e:
    print "ERROR"
    sys.exit(e)
