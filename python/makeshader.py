#!/usr/bin/env python

import sys
import numpy as np

infile = open(sys.argv[1], 'r')

header = infile.readline().rstrip()
n = int(header)

tpi = 2*np.pi

lb = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ub = np.array([ 1.0,  1.0, tpi, tpi, 4.0, 4.0, 2.0, 2.0])
rng = ub-lb
scl = rng/511.0
order = 'uvrpltsh'
tol = 1e-4

#print '    // const vec4 scl = vec4({:}, {:}, {:}, {:});'.format(*list(scl[np.arange(0,8,2)]))

for i in range(n):
    line = infile.readline().rstrip()
    nums = np.array(map(float, line.split()))
    for j in range(8):
        if nums[j] < lb[j]-tol or nums[j] > ub[j]+tol:
            print '{} is not in range [{},{}] with value {}'.format(
                order[j], lb[j], ub[j], nums[j])
            sys.exit(1)
        nums[j] = max(nums[j], lb[j])
        nums[j] = min(nums[j], ub[j])
    '''
    print 'n=', nums
    print 'lb=', lb
    print 'ub=', ub
    print nums >= lb
    print nums <= ub
    '''
    assert(np.all(nums >= lb) and np.all(nums <= ub))
    u = np.round(511*(nums - lb)/rng)
    outputs = []
    for j in range(4):
        na = u[2*j + 0]
        nb = u[2*j + 1]
        n = int(512*na + nb)
        assert(n % 512 == nb)
        assert(n / 512 == na)
        outputs.append(n)
    print '    k += gabor(p, vec4({:}.,{:}.,{:}.,{:}.));'.format(*outputs)
    

