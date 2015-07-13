#ifndef _GABOR_H_
#define _GABOR_H_

#include <math.h>
#include <stdlib.h>
#include <cv.h>
#include <assert.h>
#include <levmar.h>

typedef cv::Mat_<unsigned char> UMat;
typedef cv::Mat_<double> DMat;

inline cv::Mat hstack(const std::vector<cv::Mat>& mats) {

  int rows = 0;
  int cols = 0;
  int type = 0;
  
  for (size_t i=0; i<mats.size(); ++i) {
    const cv::Mat& m = mats[i];
    if (i == 0) {
      rows = m.rows;
      type = m.type();
    } else {
      assert(m.rows == rows);
      assert(m.type() == type);
    }
    cols += m.cols;
  }

  cv::Mat rval(rows, cols, type);
  cols = 0;

  for (size_t i=0; i<mats.size(); ++i) {
    const cv::Mat& m = mats[i];
    m.copyTo(rval.colRange(cols, cols+m.cols));
    cols += m.cols;
  }

  return rval;

}

enum ScaleType {
  SCALE_ALWAYS,
  SCALE_REDUCE_ONLY,
  SCALE_ENLARGE_ONLY
};

inline bool scale_size(const cv::Size& sz, int dim, ScaleType type,
                       cv::Size& result) {

  int odim = std::max(sz.width, sz.height);


  if ((type == SCALE_REDUCE_ONLY && odim < dim) ||
      (type == SCALE_ENLARGE_ONLY && odim > dim) ||
      (odim == dim)) {
    result = sz;
    return false;
  }

  double f = double(dim)/odim;

  result.width = sz.width * f;
  result.height = sz.height * f;

  return true;

}
                    

inline double max_element(const cv::Mat& m) {
  double rval;
  cv::minMaxLoc(m, NULL, &rval, NULL, NULL);
  return rval;
}

inline void gencoords(const cv::Size& sz,
                      DMat& xc,
                      DMat& yc,
                      double f=1.0) {


  xc.create(sz);
  yc.create(sz);

  double fx, fy;

  const int w = sz.width;
  const int h = sz.height;

  if (w >= h) {
    fx = f;
    fy = f * double(h)/w;
    assert(fx >= fy);
  } else {
    fy = f;
    fx = f * double(w)/h;
    assert(fy >= fx);
  }
    
  for (int y=0; y<h; ++y) {
    for (int x=0; x<w; ++x) {
      xc(y,x) = (x-0.5*w)*2.0*fx/w;
      yc(y,x) = (y-0.5*h)*2.0*fy/h;
    }
  }

}

inline double weighted_error(const DMat& target,
                      const DMat& output, 
                      const DMat& wmat,
                      DMat& err) {

  err = target - output;
  if (!wmat.empty()) { err = err.mul(wmat); }
  err = err.mul(err);
  return 0.5*cv::sum(err)[0];

}

enum {
  GABOR_PARAM_U = 0,
  GABOR_PARAM_V,
  GABOR_PARAM_R,
  GABOR_PARAM_L,
  GABOR_PARAM_S,
  GABOR_PARAM_T,
  GABOR_PARAM_P,
  GABOR_PARAM_H,
  GABOR_NUM_PARAMS
};


inline void gabor_random_params(double* params,
                                cv::RNG& rng = cv::theRNG(),
                                double px_size=0.05) {

  params[GABOR_PARAM_U] = rng.uniform(-1.0, 1.0);
  params[GABOR_PARAM_V] = rng.uniform(-1.0, 1.0);
  params[GABOR_PARAM_R] = rng.uniform(0.0, 2.0*M_PI);
  params[GABOR_PARAM_L] = rng.uniform(2.5*px_size, 2.0);
  params[GABOR_PARAM_S] = rng.uniform(px_size, 1.0);
  params[GABOR_PARAM_T] = rng.uniform(px_size, 1.0);
  params[GABOR_PARAM_P] = rng.uniform(0.0, 2*M_PI);
  params[GABOR_PARAM_H] = rng.uniform(0.1, 1.5);

}

template <class T>
inline double gabor(const T* const params, size_t n,
                    const double* x, const double* y, 
                    const double* W, const double* target,
                    T* result,
                    T* jacobian=0) {

  const T& u = params[GABOR_PARAM_U];
  const T& v = params[GABOR_PARAM_V];
  const T& r = params[GABOR_PARAM_R];
  const T& l = params[GABOR_PARAM_L];
  const T& s = params[GABOR_PARAM_S];
  const T& t = params[GABOR_PARAM_T];
  const T& p = params[GABOR_PARAM_P];
  const T& h = params[GABOR_PARAM_H];

  T cr = cos(r);
  T sr = sin(r);

  T f = T(2.0*M_PI)/l;
  T s2 = s*s;
  T t2 = t*t;

  size_t joffs = 0;

  double rval = 0;

  for (size_t i=0; i<n; ++i) {


    T xp = T(x[i])-u;
    T yp = T(y[i])-v;

    T b1 = cr*xp + sr*yp;
    T b2 = -sr*xp + cr*yp;

    T b12 = b1*b1;
    T b22 = b2*b2;

    T w = exp(-b12/(T(2.0)*s2) - b22/(T(2.0)*t2));

    T k = f*b1 + p;
    T ck = cos(k);
    T o = h * ck;

    T Wi = W ? W[i] : 1.0;
    T Ti = target ? target[i] : 0.0;

    T ri = Wi * (w * o - Ti);

    rval += ri * ri;

    if (result) {
      result[i] = ri;
    }

    if (jacobian) {

      T dw_db1 = -w * b1 / (s*s);
      T dw_db2 = -w * b2 / (t*t);

      T db1_du = -cr;
      T db1_dv = -sr;
      T db1_dr = b2;

      T db2_du = sr;
      T db2_dv = -cr;
      T db2_dr = -b1;

      T dw_du = dw_db1 * db1_du + dw_db2 * db2_du;
      T dw_dv = dw_db1 * db1_dv + dw_db2 * db2_dv;
      T dw_dr = dw_db1 * db1_dr + dw_db2 * db2_dr;
      T dw_ds = w * b12 / (s2*s);
      T dw_dt = w * b22 / (t2*t);

      T dk_db1 = f;
      T dk_dp = 1;
      T dck = -sin(k);
      
      T do_db1 = h * dck * dk_db1;

      T do_dp = h * dck * dk_dp;
      T do_dh = ck;

      T do_du = do_db1 * db1_du;
      T do_dv = do_db1 * db1_dv;
      T do_dr = do_db1 * db1_dr;
      T do_dl = -do_db1 * b1 /l;

      T dg_du = Wi * (dw_du * o + w * do_du);
      T dg_dv = Wi * (dw_dv * o + w * do_dv);
      T dg_dr = Wi * (dw_dr * o + w * do_dr);
      T dg_dl = Wi * (w * do_dl);
      T dg_ds = Wi * (dw_ds * o);
      T dg_dt = Wi * (dw_dt * o);
      T dg_dp = Wi * (w * do_dp);
      T dg_dh = Wi * (w * do_dh);

      jacobian[joffs+GABOR_PARAM_U] = dg_du;
      jacobian[joffs+GABOR_PARAM_V] = dg_dv;
      jacobian[joffs+GABOR_PARAM_R] = dg_dr;
      jacobian[joffs+GABOR_PARAM_L] = dg_dl;
      jacobian[joffs+GABOR_PARAM_S] = dg_ds;
      jacobian[joffs+GABOR_PARAM_T] = dg_dt;
      jacobian[joffs+GABOR_PARAM_P] = dg_dp;
      jacobian[joffs+GABOR_PARAM_H] = dg_dh;

      joffs += GABOR_NUM_PARAMS;

    }

  }

  return 0.5 * rval;

}

inline void gabor_func(double* p, double* hx, int m, int n, void* adata);
inline void gabor_jacf(double* p, double*j, int m, int n, void* adata);

struct GaborData {

  const size_t n;

  const double* x;
  const double* y;
  const double* W;
  const double* target;

  GaborData(): n(0), x(0), y(0), W(0), target(0) {}

  GaborData(size_t nn, 
            const double* xx,
            const double* yy, 
            const double* WW,
            const double* tt): n(nn), x(xx), y(yy), W(WW), target(tt) {}

  double operator()(const double* p, double* hx=NULL, double* j=NULL) const {
    return gabor(p, n, x, y, W, target, hx, j);
  }
  
  double fit(double* params, int max_iter,
             double info[LM_INFO_SZ], double* work, double px_size=0) const {

    double* null_target = 0;
    double* null_opts = 0;
    double* null_cov = 0;
    void* adata = const_cast<GaborData*>(this);

    if (!px_size) {

      dlevmar_der(gabor_func, gabor_jacf, 
                  params, null_target,
                  GABOR_NUM_PARAMS, n,
                  max_iter, null_opts, info,
                  work, null_cov, adata);

    } else {

      double* null_ub = 0;
      double* null_dscl = 0;
      
      double lb[GABOR_NUM_PARAMS];

      for (int i=0; i<GABOR_NUM_PARAMS; ++i) {
        if (i == GABOR_PARAM_L) {
          lb[i] = px_size*2.5;
        } else if (i == GABOR_PARAM_S || i == GABOR_PARAM_T) {
          lb[i] = px_size;
        } else {
          lb[i] = -DBL_MAX;
        }
      }

      dlevmar_bc_der(gabor_func, gabor_jacf,
                     params, null_target,
                     GABOR_NUM_PARAMS, n,
                     lb, null_ub, null_dscl, 
                     max_iter, null_opts,
                     info, work, null_cov, adata);

    }

    return 0.5*info[1];

  }

};

inline void gabor_func(double* p, double* hx, int m, int n, void* adata) {
  assert(m == GABOR_NUM_PARAMS);
  const GaborData* info = static_cast<const GaborData*>(adata);
  assert(info->n == n);
  (*info)(p, hx, NULL);
}

inline void gabor_jacf(double* p, double*j, int m, int n, void* adata) {
  assert(m == GABOR_NUM_PARAMS);
  const GaborData* info = static_cast<const GaborData*>(adata);
  assert(info->n == n);
  (*info)(p, NULL, j);
}

#endif
