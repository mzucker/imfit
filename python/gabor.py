import cv2
import numpy as np
import scipy.optimize
import sys

from ImageBases import *

######################################################################
# Verify Jacobians work

def verify_jacobians(model, params, x, y, guess, weights):

    target, J = model.compute(params, x, y, jac=True)

    Jn = np.zeros_like(J)
    delta = np.zeros_like(params)
    h = 1e-5

    for i in range(len(params)):
        delta[i] = h
        op = model.compute(params+delta, x, y, False)
        on = model.compute(params-delta, x, y, False)
        delta[i] = 0
        Jn[:, i] = (op-on).flatten()/(2*h)

    maxerr = np.abs(J-Jn).max(axis=0)
    print 'max err in J:', maxerr

    assert( maxerr.max() < 1e-5 )

    errsq, g = model.error(guess, target, x, y, True, weights)

    gn = np.zeros_like(g)

    delta = np.zeros_like(params)
    h = 1e-5
    for i in range(guess.size):
        delta[i] = h
        ep = model.error(guess+delta, target, x, y, False, weights)
        en = model.error(guess-delta, target, x, y, False, weights)
        gn[i] = (ep - en)/(2*h)
        delta[i] = 0


    gerr = np.abs(g - gn)
    print 'g err:', gerr
    assert(gerr.max() < 1e-4)

    arrays = [ normalize(target) ]

    for i in range(J.shape[1]):
        arrays.append( normalize( J[:,i].reshape(x.shape) ) )

    display = np.hstack(tuple(arrays))

    win = 'window'
    cv2.namedWindow(win)
    cv2.imshow(win, display)
    while cv2.waitKey(5) < 0: pass

models = [ GaborModel() ]

fitme = SumModel(models)

xyrng = np.linspace(-1, 1, 51)
x, y = np.meshgrid(xyrng, xyrng)

params = fitme.random_params()
params[GABOR_PARAM_U] = 0
params[GABOR_PARAM_V] = 0
# so clamp S to be min of 0.25, max of 4

guess = fitme.mutate_params(params)

guess = 0.3*guess + 0.7*params

weights = np.exp(-(x**2 + y**2)/(2*0.25))

verify_jacobians(fitme, params, x, y, guess, weights)
#sys.exit(0)

orig = fitme.compute(guess, x, y, False)
target = fitme.compute(params, x, y, False)

res = scipy.optimize.minimize(fitme.objective(target, x, y, weights=weights),
                              guess, jac=True,
                              method='L-BFGS-B')

print res

fp = res.x

final = fitme.compute(fp, x, y, False)

error = np.abs(final - target)

display = np.hstack(tuple([normalize(x) for x in [orig, target, final]]))
display = np.hstack((display, error))

win = 'window'
cv2.namedWindow(win)
cv2.imshow(win, display)
while cv2.waitKey(5) < 0: pass

