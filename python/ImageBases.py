import numpy as np

GABOR_PARAM_U = 0
GABOR_PARAM_V = 1
GABOR_PARAM_R = 2
GABOR_PARAM_L = 3
GABOR_PARAM_S = 4
GABOR_PARAM_P = 5
GABOR_PARAM_H = 6
GABOR_NUM_PARAMS = 7

######################################################################

class GeneralModel:

    def error(self, params, target, x, y, jac=True, weights=None):

        target = target.flatten()
        if weights is None:
            weights = 1
        else:
            weights = weights.flatten()

        res = self.compute(params, x, y, jac)

        if jac:
            o, J = res
        else:
            o = res

        o = o.flatten()

        diff = weights * (target - o)
        rval = 0.5 * np.dot(diff, diff)

        if jac:
            g = -np.dot(weights * diff, J)
            return rval, g
        else:
            return rval

    def objective(self, target, x, y, jac=True, weights=None):
        return lambda p: self.error(p, target, x, y, 
                                    jac=jac, weights=weights)
        

######################################################################

class ConstantModel(GeneralModel):

    def __init__(self):
        self.n = 1
        self.name = 'constant'

    def random_params(self):
        
        rval = np.empty(1)
        rval[0] = np.random.uniform(-1, 1)

        return rval

    def mutate_params(self, params):

        rval = params.copy()
        rval[0] += np.random.uniform(-0.5, 0.5)

        return rval

    def compute(self, params, x, y, jac=True):

        assert(params.size == 1)
        assert(x.shape == y.shape)

        scl = 1

        rval = np.ones_like(x) * params[0] * scl

        if not jac:
            return rval

        J = np.ones((x.size, 1)) * scl

        return rval, J

######################################################################

class GaborModel(GeneralModel):

    def __init__(self):
        self.n = GABOR_NUM_PARAMS
        self.name = 'gabor'

    def random_params(self):

        rval = np.empty(GABOR_NUM_PARAMS)
        rval[GABOR_PARAM_U] = np.random.uniform(-1, 1)
        rval[GABOR_PARAM_V] = np.random.uniform(-1, 1)
        rval[GABOR_PARAM_R] = np.random.uniform(0, 2*np.pi)
        rval[GABOR_PARAM_L] = np.random.uniform(0.5, 2.0)
        rval[GABOR_PARAM_S] = np.random.uniform(0.25, 1.0)
        rval[GABOR_PARAM_P] = np.random.uniform(0, 2*np.pi)
        rval[GABOR_PARAM_H] = np.random.uniform(0.1, 1.5)

        return rval

    def mutate_params(self, params, 
                      xy_std=0.3, r_std=0.3, 
                      l_frac=0.125, s_frac=0.125, 
                      p_std=0.3, h_frac=0.125):

        rval = params.copy()

        rval[GABOR_PARAM_U] += np.random.normal(scale=xy_std)
        rval[GABOR_PARAM_V] += np.random.normal(scale=xy_std)
        rval[GABOR_PARAM_R] += np.random.normal(scale=r_std)
        rval[GABOR_PARAM_L] *= 1 + np.random.normal(scale=l_frac)
        rval[GABOR_PARAM_S] *= 1 + np.random.normal(scale=s_frac)
        rval[GABOR_PARAM_P] += np.random.normal(scale=p_std)
        rval[GABOR_PARAM_H] *= 1 + np.random.normal(h_frac)

        return rval

    ######################################################################
    # Gabor filter / function + derivatives
    #
    # params should be a vector of length 7 with elements:
    #   - u: x center of Gabor filter
    #   - v: y center of Gabor filter
    #   - r: rotation of Gabor filter
    #   - l: wavelength of sinusoidal component
    #   - s: multiplier on l to get sigma for Gaussian component in [0.25, 1]
    #   - p: phase of Gabor filter
    #   - h: amplitude
    #   - t: (TODO) multiplier on s to get width in cross direction

    def compute(self, params, x, y, jac=True):

        assert(params.size == GABOR_NUM_PARAMS)
        assert(x.shape == y.shape)

        u = params[GABOR_PARAM_U]
        v = params[GABOR_PARAM_V]
        r = params[GABOR_PARAM_R]
        l = params[GABOR_PARAM_L]
        s = params[GABOR_PARAM_S]
        p = params[GABOR_PARAM_P]
        h = params[GABOR_PARAM_H]

        cr = np.cos(r)
        sr = np.sin(r)

        xp = x-u
        yp = y-v

        r2 = xp**2 + yp**2

        f = 2*np.pi/l

        sl = s * l
        w = np.exp(-r2/(2*sl**2))

        cxsy = cr*xp + sr*yp
        a = f*cxsy

        ca = np.cos(a + p)

        o = h * ca

        g = o * w

        if not jac:
            return g

        df_dl = -f/l

        dw_du = w * xp / sl**2
        dw_dv = w * yp / sl**2
        dw_dl = w * r2 * s / sl**3
        dw_ds = w * r2 * l / sl**3 

        da_du = -f*cr
        da_dv = -f*sr
        da_dr = f*(-sr*xp + cr*yp)
        da_dl = cxsy * df_dl

        sa = np.sin(a + p)
        do = -h * sa

        do_du = do * da_du
        do_dv = do * da_dv
        do_dr = do * da_dr
        do_dl = do * da_dl
        do_dp = do
        do_dh = ca

        dg_du = dw_du * o + w * do_du
        dg_dv = dw_dv * o + w * do_dv
        dg_dr = w * do_dr
        dg_dl = dw_dl * o + w * do_dl 
        dg_ds = dw_ds * o 
        dg_dp = w * do_dp
        dg_dh = w * do_dh

        J = np.zeros((x.size, params.size))

        J[:,GABOR_PARAM_U] = dg_du.flatten()
        J[:,GABOR_PARAM_V] = dg_dv.flatten()
        J[:,GABOR_PARAM_R] = dg_dr.flatten()
        J[:,GABOR_PARAM_L] = dg_dl.flatten()
        J[:,GABOR_PARAM_S] = dg_ds.flatten()
        J[:,GABOR_PARAM_P] = dg_dp.flatten()
        J[:,GABOR_PARAM_H] = dg_dh.flatten()

        return g, J

######################################################################

class SumModel(GeneralModel):

    def __init__(self, models):
        self.n = sum([model.n for model in models])
        self.name = 'sum(' + ','.join([model.name for model in models]) + ')'
        self.models = models
        
    def random_params(self):
        
        rval = np.empty(self.n)

        i = 0
        for model in self.models:
            rval[i:i+model.n] = model.random_params()
            i += model.n
            
        assert(i == self.n)

        return rval

    def mutate_params(self, params):

        rval = np.empty_like(params)

        i = 0
        for model in self.models:
            rval[i:i+model.n] = model.mutate_params(params[i:i+model.n])
            i += model.n

        assert(i == self.n)

        return rval

    def compute(self, params, x, y, jac=True):

        assert(params.size == self.n)
        assert(x.shape == y.shape)

        if jac: 
            J = np.empty((x.size, params.size))

        output = np.zeros_like(x)

        i = 0

        for model in self.models:
            res = model.compute(params[i:i+model.n], x, y, jac)
            if jac:
                model_output, model_J = res
                J[:, i:i+model.n] = model_J
            else:
                model_output = res
            output += model_output
            i += model.n

        assert(i == self.n)

        if not jac:
            return output
        else:
            return output, J
        
######################################################################

def normalize(v):
    vmax = np.abs(v).max()
    if not vmax:
        return 0.5*np.ones_like(v)
    else:
        return 0.5 + 0.5*v/vmax
