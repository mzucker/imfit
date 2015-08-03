#!/usr/bin/env python

import sys
import numpy as np

infile = open(sys.argv[1], 'r')

header = infile.readline().rstrip()
n = int(header)

alldata = []

for i in range(n):
    line = infile.readline().rstrip()
    nums = line.split()
    assert( len(nums) == 8 )
    alldata.append(map(float, nums))
    #print '    k += gabor(uv, vec4({:}, {:}, {:}, {:}), vec4({:}, {:}, {:}, {:}));'.format(*nums)


A = np.array(alldata)
amin = A.min(axis=0)
amax = A.max(axis=0)
print amin
print amax
    
