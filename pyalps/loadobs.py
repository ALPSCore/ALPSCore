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

import pyalps as alps
import pyalps.hdf5 as h5
import pyalps.alea as alea

iar = h5.iArchive('loadobs.h5')

for name in iar.list_children('/simulation/results'):
    
    print name
    
    print '/simulation/results/' + alps.hdf5_name_decode(name) + '/mean/value'
    
    if iar.is_scalar('/simulation/results/' + alps.hdf5_name_encode(name) + '/mean/value'):
        obs = alea.MCScalarData()
    else:
        obs = alea.MCVectorData()
    obs.load('loadobs.h5', '/simulation/results/' + alps.hdf5_name_encode(name))
    print name + ": " + str(obs)
