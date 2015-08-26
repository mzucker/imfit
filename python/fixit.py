#!/usr/bin/env python

import sys
import numpy as np

infile = open(sys.argv[1], 'r')

header = infile.readline().rstrip()
n = int(header)

print n


oldorder = 'uvrlstph'
neworder = 'uvrpltsh'
old_lb = dict(zip(oldorder, [-1.0, -1.0, -np.pi, 0.0, 0.0, 0.0, -np.pi, -2.0]))
old_ub = dict(zip(oldorder, [ 1.0,  1.0,  np.pi, 4.0, 2.0, 2.0,  np.pi,  2.0]))

tpi = 2*np.pi

new_lb = dict(zip(neworder, [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
new_ub = dict(zip(neworder, [ 1.0,  1.0, tpi, tpi, 4.0, 4.0, 2.0, 2.0]))

for i in range(n):

    line = infile.readline().rstrip()

    nums = map(float, line.split())
    assert( len(nums) == len(oldorder) )

    d = dict(zip(oldorder, nums))

    for var, n in d.iteritems():
        assert(n >= old_lb[var] and n <= old_ub[var])

    if d['h'] < 0.0:
        d['h'] *= -1
        d['p'] += np.pi

    if d['p'] < 0.0:
        d['p'] += tpi

    if d['r'] < 0.0:
        d['r'] += tpi

    for var, n in d.iteritems():
        assert(n >= new_lb[var] and n <= new_ub[var])

    print ' '.join([str(d[var]) for var in neworder])




