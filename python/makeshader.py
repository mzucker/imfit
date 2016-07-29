#!/usr/bin/env python

'''Script to encode the list of 8-tuples emitted by the imfit program
and format them for use in the shader. The order emitted by imfit is:

  u: center x-coordinate of Gabor basis function
  v: center y-coordinate
  r: rotation angle of Gabor basis function
  p: phase of basis function
  l: wavelength (inverse frequency) of Gabor function
  t: width perpendicular to sinusoidal component
  s: width along sinusoidal component
  h: amplitude

The eight numbers are rescaled and quantized into the range [0, 511]
and encoded into four floating-point numbers (uv, rp, lt, sh).

'''

import sys
import numpy as np

if len(sys.argv) != 2:
    print 'usage:', sys.argv[0], 'params.txt'
    sys.exit(0)

infile = open(sys.argv[1], 'r')

count_line = infile.readline().rstrip()
num_functions = int(count_line)

two_pi = 2*np.pi

lower_bound = np.array([-1.0, -1.0, 0.0,    0.0,    0.0, 0.0, 0.0, 0.0])
upper_bound = np.array([ 1.0,  1.0, two_pi, two_pi, 4.0, 4.0, 2.0, 2.0])

ranges = upper_bound-lower_bound

var_names = 'uvrpltsh'
tol = 1e-4

for i in range(num_functions):

    line = infile.readline().rstrip()
    
    nums = np.array(map(float, line.split()))
    
    for j in range(8):

        if nums[j] < lower_bound[j]-tol or nums[j] > upper_bound[j]+tol:
            print '{} is not in range [{},{}] with value {}'.format(
                var_names[j], lower_bound[j], upper_bound[j], nums[j])
            sys.exit(1)

    nums = np.clip(nums, lower_bound, upper_bound)
    assert(np.all(nums >= lower_bound) and np.all(nums <= upper_bound))
    
    u = np.round(511*(nums - lower_bound)/ranges).astype(int)
    
    outputs = []
    
    for j in range(4):
        na = u[2*j + 0]
        nb = u[2*j + 1]
        n = 512*na + nb
        assert(n % 512 == nb)
        assert(n / 512 == na)
        outputs.append(n)
        
    print '    k += gabor(p, vec4({:}.,{:}.,{:}.,{:}.));'.format(*outputs)
