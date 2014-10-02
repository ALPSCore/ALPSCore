# Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
# All rights reserved. Use is subject to license terms. See LICENSE.TXT
# For use in publications, see ACKNOWLEDGE.TXT

import pyalps.hdf5 as hdf5
import pyalps.ngs as ngs
import sys, time, getopt

import ising_c as ising

if __name__ == '__main__':

    try:
        optlist, positional = getopt.getopt(sys.argv[1:], 'T:c')
        args = dict(optlist)
        try:
            limit = float(args['-T'])
        except KeyError:
            limit = 0
        resume = True if 'c' in args else False
        outfile = positional[0]
    except (IndexError, getopt.GetoptError):
        print 'usage: [-T timelimit] [-c] outputfile'
        exit()

    sim = ising.sim(ngs.params({
        'L': 100,
        'THERMALIZATION': 1000,
        'SWEEPS': 10000,
        'T': 2
    }))

    if resume:
        try:
            with hdf5.archive(outfile[0:outfile.rfind('.h5')] + '.clone0.h5', 'r') as ar:
                sim.load(ar['/'])
        except ArchiveNotFound: pass

    if limit == 0:
        sim.run(lambda: False)
    else:
        start = time.time()
        sim.run(lambda: time.time() > start + float(limit))

    with hdf5.archive(outfile[0:outfile.rfind('.h5')] + '.clone0.h5', 'w') as ar:
        ar['/'] = sim

    results = sim.collectResults() # TODO: how should we do that?
    print results

    with hdf5.archive(outfile, 'w') as ar: # TODO: how sould we name archive? ngs.hdf5.archive?
        ar['/parameters'] = sim.parameters
        ar['/simulation/results'] = results
