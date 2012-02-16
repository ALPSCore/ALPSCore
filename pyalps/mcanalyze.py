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

#iar = h5.iArchive('test/scalartestfile.h5')
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


#iar = h5.iArchive('test/scalartestfile.h5')
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
