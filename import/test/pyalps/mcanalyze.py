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



import numpy
import pyalps
import mcanalyze
import matplotlib.pyplot as plt
import pyalps.plot as alpsplt
import pyalps.hdf5 as h5



print 
print "********** MCANALYZE PYTHON TEST **************"
print 

rng = pyalps.pytools.rng(42)

tmp1 = 1
tmp2 = 1
DATA = []

tmp1 = 1
for i in range(10000):
  tmp1 = 0.9 * tmp1 + 0.1 * (2*rng()-1)
  DATA = DATA + [tmp1]

NPDATA = numpy.array(DATA)

#TS = mcanalyze.MCScalarTimeseries(NPDATA)
#Auto_corr = mcanalyze.autocorrelation_range(TS, mcanalyze.size(TS)-1)

#binning_error = mcanalyze.binning_error(TS)
#uncorrelated_error = mcanalyze.uncorrelated_error(TS)

#expo_corr_time = mcanalyze.exponential_autocorrelation_time_decay(Auto_corr, 1, 0.2)
#int_corr_time = mcanalyze.integrated_autocorrelation_time_decay(Auto_corr, expo_corr_time, 0.8)

#print "Size: " + str(mcanalyze.size(TS))
#print "Mean: " + str(mcanalyze.mean(TS))
#print "Variance: " + str(mcanalyze.variance(TS))
#print "integrated_corr_time: " + str(int_corr_time)
#print "binning_error: " + str(binning_error)
#print "uncorrelated_error: " + str(uncorrelated_error)




#auto = mcanalyze.autocorrelation_decay(TS,0.001)
#t = numpy.arange(0.0, mcanalyze.size(auto), 0.5)
#fit = mcanalyze.exponential_autocorrelation_time_range(auto, 1, mcanalyze.size(auto)/3)
#plt.plot(t, fit.first * numpy.exp(fit.second * t))
#alpsplt.plot(mcanalyze.make_dataset(auto))
#plt.show()





print
print "before"
print

#iar = h5.archive('test/scalartestfile.h5', 'r')
#for name in iar.list_children('/simulation/results'):
#    if iar.is_scalar('/simulation/results/' + pyalps.hdf5_name_encode(name) + '/mean/value'):
#        obs = pyalps.alea.MCScalarData()
#    else:
#        obs = pyalps.alea.MCVectorData()
#    obs.load('test/scalartestfile.h5', '/simulation/results/' + pyalps.hdf5_name_encode(name))
#    print name + ": " + str(obs)

#del iar

#print mcanalyze.mean(obs)

#mcanalyze.write_to_file('test/scalartestfile.h5', "E", "/mean/error", binning_error)

print
print "after"
print


#iar = h5.archive('test/scalartestfile.h5', 'r')
#for name in iar.list_children('/simulation/results'):
#    if iar.is_scalar('/simulation/results/' + pyalps.hdf5_name_encode(name) + '/mean/value'):
#        obs = pyalps.alea.MCScalarData()
#    else:
#        obs = pyalps.alea.MCVectorData()
#    obs.load('test/scalartestfile.h5', '/simulation/results/' + pyalps.hdf5_name_encode(name))
#    print name + ": " + str(obs)


print 
print "********** TEST END **************"
print 
