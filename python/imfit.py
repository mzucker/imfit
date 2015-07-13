#!/usr/bin/env python

import cv2
import numpy as np
import ImageBases as ib
import scipy.optimize
import sys
import argparse

def load_image(string):
    image = cv2.imread(string, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    if image is None or len(image) == 0:
        raise argparse.ArgumentTypeError('error loading image:' + string)
    return image

parser = argparse.ArgumentParser(description='Fit images with Gabor functions')

parser.add_argument('image', metavar='IMGFILE',
                    type=load_image, nargs=1,
                    help='source image file')

parser.add_argument('-w --weights', metavar='IMGFILE',
                    dest='weights', 
                    type=load_image, default=None,
                    help='weight image file')

parser.add_argument('-i, --input', type=argparse.FileType('r'), 
                    nargs=1, dest='input',
                    help='load previous output', 
                    metavar='DATAFILE')

parser.add_argument('-a, --action', type=str, nargs=1, 
                    metavar='ACTION', dest='action',
                    choices=['greedyfit', 'refine'], 
                    default='refine',
                    help='action to take (default greedyfit)')

parser.add_argument('-n, --num-models', type=int, 
                    metavar='N', dest='num_models',
                    default=150, help='number of models to fit')

parser.add_argument('-f, --num-fits', type=int, 
                    metavar='N', dest='num_fits',
                    default=100, help='number of random starts per model')

parser.add_argument('-m, --maxiter', type=int, 
                    metavar='N', dest='maxiter',
                    default=100, help='max # of function evals per trial')

parser.add_argument('-r, --refine', type=int, 
                    metavar='N', dest='refine',
                    default=None, help='max # of function evals to refine')
 
parser.add_argument('-s, --max-size', type=int, 
                    metavar='N', dest='maxsize',
                    default=100, help='maximum width/height of input image')

parser.add_argument('-p, --preview-size', type=int, 
                    metavar='N', dest='psize',
                    default=256, help='size of preview image')

parser.add_argument('-g, --no-gui', dest='gui', action='store_const',
                    const=False, default=True, help='suppress graphical output')
                    


args = parser.parse_args()

num_models = args.num_models
num_fits = args.num_fits
maxiter = args.maxiter
refine = args.refine
gray = args.image[0]
gui = args.gui

weights = args.weights

if weights is not None:
    assert(weights.shape == gray.shape)
    weights = weights.astype('float')
    weights /= weights.max()

##################################################
# set up images and xy ranges

h, w = gray.shape

sz = max(h, w)
if sz > args.maxsize:
    f = float(args.maxsize) / sz
    gray = cv2.resize(gray, (0,0), None, f, f, interpolation=cv2.INTER_AREA)
    if weights is not None:
        weights = cv2.resize(weights, (0,0), None, f, f, interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    sz = max(h, w)

if sz == h:
    fy = 1
    fx = float(w)/h
    big_h = args.psize
    big_w = round(big_h*fx)
    assert(fy >= fx)
    assert(big_h >= big_w)
else:
    fx = 1
    fy = float(h)/w
    big_w = args.psize
    big_h = round(big_w*fy)
    assert(fx >= fy)
    assert(big_w >= big_h)

img = gray.astype(float)
img -= img.mean()
img /= np.abs(img).max()

xrng = np.linspace(-fx, fx, img.shape[1])
yrng = np.linspace(-fy, fy, img.shape[0])

h = yrng[1]-yrng[0]

x, y = np.meshgrid(xrng, yrng)

bigxrng = np.linspace(-fx, fx, big_w)
bigyrng = np.linspace(-fy, fy, big_h)
bigx, bigy = np.meshgrid(bigxrng, bigyrng)

##################################################
# set up models

model = ib.GaborModel()

if weights is None:
    wscl = 1
else:
    wscl = weights

output = np.zeros_like(img)
error = wscl*(img - output)
errsum = 0.5*(error**2).sum()
print 'initial error:', errsum

big_output = np.zeros_like(bigx)

swin = 'Progress'
pwin = 'Preview'

if gui:
    cv2.namedWindow(swin)
    cv2.imshow(swin, np.hstack((img, output, np.abs(error))))
    cv2.imshow(pwin, big_output)
    cv2.waitKey(5)

bounds = [(None, None)] * ib.GABOR_NUM_PARAMS
bounds[ib.GABOR_PARAM_S] = (0.25, 1.0)
bounds[ib.GABOR_PARAM_L] = (3.0*h, None)

models = []
param_vecs = []

for i in range(num_models):
    
    best_res = None

    for fit in range(num_fits):
        sys.stdout.write('.')
        sys.stdout.flush()
        guess = model.random_params()

        guess[ib.GABOR_PARAM_U] = np.random.uniform(-fx, fx)
        guess[ib.GABOR_PARAM_V] = np.random.uniform(-fy, fy)

        # TODO: skip if maxiter = 0
        res = scipy.optimize.minimize(
            model.objective(error, x, y, weights=weights),
            guess, jac=True, bounds=bounds,
            method='L-BFGS-B', options=dict(maxiter=maxiter))

        if best_res is None or res.fun < best_res.fun:
            tmp_output = output + model.compute(res.x, x, y, False)
            tmp_error = wscl * (img - tmp_output)
            edisplay = np.abs(tmp_error)
            edisplay = 1-np.exp(-10*edisplay)
            if gui:
                cv2.imshow(swin, np.hstack((img, tmp_output, edisplay)))
                cv2.waitKey(5)
            best_res = res

    print

    if refine is None or refine > 0:
        if refine is None:
            odict = None
        else:
            odict = dict(maxiter=refine)
        best_res = scipy.optimize.minimize(
            model.objective(error, x, y, weights=weights),
            best_res.x, jac=True, bounds=bounds,
            method='L-BFGS-B', options=odict)


    newoutput = output + model.compute(best_res.x, x, y, False)
    newerror = wscl * (img - newoutput)
    newerrsum = 0.5*(newerror**2).sum()
    print 'error after {}: {}'.format(i+1, newerrsum)

    if newerrsum > errsum:
        print 'skipping iteration because error increased!'

    else:

        output = newoutput
        error = newerror
        errsum = newerrsum

        models.append(ib.GaborModel())
        param_vecs.append(best_res.x)

        if gui:
            big_output += model.compute(best_res.x, bigx, bigy, False)
            edisplay = np.abs(error)
            edisplay = 1-np.exp(-10*edisplay)

            cv2.imshow(swin, np.hstack((img, output, edisplay)))
            cv2.imshow(pwin, big_output)
            cv2.waitKey(5)

params = np.hstack(tuple(param_vecs))
model = ib.SumModel(models)
bounds = bounds * len(models)

if gui:

    big_output = model.compute(params, bigx, bigy, False)

    cv2.imshow(pwin, big_output)
    init = model.objective(img, x, y, weights=weights, jac=False)(params)
    #print 'init is', init, 'hit a key to optimize.'
    while cv2.waitKey(5) < 0: pass

'''

res = scipy.optimize.minimize(model.objective(img, x, y, weights=weights),
                              params, jac=True, bounds=bounds,
                              method='L-BFGS-B')

final = res.fun
params = res.x
big_output = model.compute(params, bigx, bigy, False)

cv2.imshow(pwin, big_output)
print 'final is', final, 'hit a key to quit.'
while cv2.waitKey(5) < 0: pass


#cv2.imshow(win, img)




'''
