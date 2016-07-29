#ifdef NDEBUG
#undef NDEBUG
#endif

#include <assert.h>

#include <math.h>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

#include <levmar.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

//////////////////////////////////////////////////////////////////////
// Typedef for byte and double-oriented matrices/images

typedef cv::Mat_<unsigned char> UMat;
typedef cv::Mat_<double> DMat;

//////////////////////////////////////////////////////////////////////
// Utility function to horizontally stack multiple matrices/images

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

//////////////////////////////////////////////////////////////////////
// Utility function to rescale image so that max(width, height) is
// equal to desired dimension.

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
                    
//////////////////////////////////////////////////////////////////////
// Convienent wrapper on cv::minMaxLoc to just return max.

inline double max_element(const cv::Mat& m) {
  double rval;
  cv::minMaxLoc(m, NULL, &rval, NULL, NULL);
  return rval;
}

//////////////////////////////////////////////////////////////////////
// Generate uv coords in range [-1,1] for an image of a given size.

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

  std::cout << "using fx=" << fx << ", fy=" << fy << "\n";
    
  for (int y=0; y<h; ++y) {
    for (int x=0; x<w; ++x) {
      xc(y,x) = (x-0.5*w)*2.0*fx/w;
      yc(y,x) = (y-0.5*h)*2.0*fy/h;
    }
  }

}

//////////////////////////////////////////////////////////////////////
// Compute weighted error and optionally rescale it for display.

inline double weighted_error(const DMat& target,
                             const DMat& output, 
                             const DMat& wmat,
                             DMat& err,
                             double display_pow=0.0) {

  err = output - target;
  if (!wmat.empty()) { err = err.mul(wmat); }
  double rval = 0.5*cv::sum(err.mul(err))[0];
  if (display_pow > 0) {
    for (int i=0; i<err.rows; ++i) {
      for (int j=0; j<err.cols; ++j) {
        err(i,j) = pow(err(i,j), display_pow);
      }
    }
  }
  return rval;

}

//////////////////////////////////////////////////////////////////////
// Set up Gabor function parameters:
//
//   var   min    max    description
//
//   u     -1     1      center x-coordinate
//   v     -1     1      center y-coordinate
//   r     0      2*pi   overall orentation/rotation
//   p     0      2*pi   phase
//   l     2.5*px 4      wavelength (inverse frequency)
//   t     px     4      width perpendicular to sinusoid
//   s     px     2      width along sinusoid
//   h     0      2      amplitude
//
// Beyond the box constraints above, we require the following
// inequalities be met:
//
//   s >= 0.03125 * l <-> s - 0.03125 * l >= 0
//   s <= 0.5 * l     <->     0.5 * l - s >= 0
//   t >= s           <->           t - s >= 0
//   t <= 8 * s       <->       8 * s - t >= 0
//
// These prevent the Gabor functions from being too long and skinny
// which tends to overfit.

enum {
  GABOR_PARAM_U = 0,  // in [-1, 1]
  GABOR_PARAM_V,      // in [-1, 1]
  GABOR_PARAM_R,      // radians in [0, 2*pi]
  GABOR_PARAM_P,      // radians in [0, 2*pi]
  GABOR_PARAM_L,      // in [2.5*px, 4.0]
  GABOR_PARAM_T,      // in [px, 4.0], we want t >= s, and basically unlimited on top
  GABOR_PARAM_S,      // in [px, 2.0], we want s >= 0.125*l and s <= 0.5*l
  GABOR_PARAM_H,      // in [0, 2.0]
  GABOR_NUM_PARAMS,
  GABOR_NUM_INEQ=4,
  GABOR_C_SIZE=GABOR_NUM_INEQ*GABOR_NUM_PARAMS
};

//////////////////////////////////////////////////////////////////////
// Gather bounds for levmar library.

inline void gabor_bounds(double px, 
                         double lb[GABOR_NUM_PARAMS], 
                         double ub[GABOR_NUM_PARAMS]) {

  if (lb) {
    lb[GABOR_PARAM_U] = -1.0;
    lb[GABOR_PARAM_V] = -1.0;
    lb[GABOR_PARAM_R] = 0.0;
    lb[GABOR_PARAM_P] = 0.0;
    lb[GABOR_PARAM_L] = 2.5*px;
    lb[GABOR_PARAM_T] = px;
    lb[GABOR_PARAM_S] = px;
    lb[GABOR_PARAM_H] = 0.0;
  }

  if (ub) {
    ub[GABOR_PARAM_U] = 1.0;
    ub[GABOR_PARAM_V] = 1.0;
    ub[GABOR_PARAM_R] = 2.0*M_PI;
    ub[GABOR_PARAM_P] = 2.0*M_PI;
    ub[GABOR_PARAM_L] = 4.0;
    ub[GABOR_PARAM_T] = 4.0;
    ub[GABOR_PARAM_S] = 2.0;
    ub[GABOR_PARAM_H] = 2.0;
  }

}

//////////////////////////////////////////////////////////////////////
// Gather inequalities for levmar library.

inline void gabor_ineq(double C[GABOR_C_SIZE],
                       double d[GABOR_NUM_INEQ]) {

  memset(C, 0, GABOR_C_SIZE*sizeof(double));
  memset(d, 0, GABOR_NUM_INEQ*sizeof(double));

  // s >= 0.03125 * l --> s - 0.03125*l >= 0
  // s <= 0.5* l    --> 0.5*l -  s  >= 0
  // t >= s         --> t - s       >= 0
  // t <= 8.0*s     --> 8.0*s - t   >= 0
  
  C[GABOR_NUM_PARAMS*0 + GABOR_PARAM_S] =  1.0;
  C[GABOR_NUM_PARAMS*0 + GABOR_PARAM_L] = -0.03125;
  
  C[GABOR_NUM_PARAMS*1 + GABOR_PARAM_S] = -1.0;
  C[GABOR_NUM_PARAMS*1 + GABOR_PARAM_L] =  0.5;

  C[GABOR_NUM_PARAMS*2 + GABOR_PARAM_S] = -1.0;
  C[GABOR_NUM_PARAMS*2 + GABOR_PARAM_T] =  1.0;

  C[GABOR_NUM_PARAMS*3 + GABOR_PARAM_S] =  8.0;
  C[GABOR_NUM_PARAMS*3 + GABOR_PARAM_T] = -1.0;

}

//////////////////////////////////////////////////////////////////////
// Generate random Gabor parameters inside box and use rejection
// sampling to verify inequality constraints.

inline void gabor_random_params(double* params, double px_size,
                                cv::RNG& rng) {

  double lb[GABOR_NUM_PARAMS], ub[GABOR_NUM_PARAMS];
  double C[GABOR_C_SIZE], d[GABOR_NUM_INEQ];

  gabor_bounds(px_size, lb, ub);
  gabor_ineq(C, d);

  DMat Cmat(GABOR_NUM_INEQ, GABOR_NUM_PARAMS, C);
  DMat dvec(GABOR_NUM_INEQ, 1, d);
  DMat pvec(GABOR_NUM_PARAMS, 1, params);

  // Hopefully this takes MUCH less than 1000 iterations.
  for (int iter=0; iter<1000; ++iter) {

    // Uniform inside box
    for (int i=0; i<GABOR_NUM_PARAMS; ++i) {
      params[i] = rng.uniform(lb[i], ub[i]);
    }

    // Remainder is rejection sampling
    bool ok = true;
    DMat rvec = Cmat*pvec - dvec;

    for (int j=0; j<GABOR_NUM_INEQ; ++j) {
      if (rvec(j) < 0) {
        ok = false;
      }
    }

    if (ok) {
      break;
    }
    
  }


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
  const T& p = params[GABOR_PARAM_P];
  const T& l = params[GABOR_PARAM_L];
  const T& t = params[GABOR_PARAM_T];
  const T& s = params[GABOR_PARAM_S];
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

    /*
    if (i == 0) {

#define DOUT(var) (std::cout << (#var) << "=" << (var) << "\n")
      DOUT(xp);
      DOUT(yp);
      DOUT(b1);
      DOUT(b2);
      DOUT(b12);
      DOUT(b22);
      DOUT(w);
      DOUT(k);
      DOUT(k);
      DOUT(ck);
      DOUT(o);
      DOUT(Wi);
      DOUT(Ti);
      DOUT(ri);
#undef DOUT
      std::cout << "\n";
    }
    */
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
      jacobian[joffs+GABOR_PARAM_P] = dg_dp;
      jacobian[joffs+GABOR_PARAM_L] = dg_dl;
      jacobian[joffs+GABOR_PARAM_T] = dg_dt;
      jacobian[joffs+GABOR_PARAM_S] = dg_ds;
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


      double lb[GABOR_NUM_PARAMS], ub[GABOR_NUM_PARAMS];
      double C[GABOR_C_SIZE], d[GABOR_NUM_INEQ];

      gabor_bounds(px_size, lb, ub);
      gabor_ineq(C, d);

      dlevmar_blic_der(gabor_func, gabor_jacf,
                       params, null_target,
                       GABOR_NUM_PARAMS, n,
                       lb, ub, 
                       C, d, GABOR_NUM_INEQ,
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


#define ERR_DISPLAY_POW 0.25

struct ImFitOptions {

  std::string image_file;  // positional

  std::string weight_file; // w
  std::string input_file;  // i 
  std::string output_file; // o

  std::string action;      // a

  int num_models;          // n
  int greedy_num_fits;     // f
  int greedy_init_iter;    // m
  int greedy_refine_iter;  // r
  int greedy_replace_iter; // R
  int max_size;            // s
  int preview_size;        // p
  int full_iter;           // F
  double full_alpha;       // A
  bool gui;                // g

  ImFitOptions() {

    char buf[1024];

    struct tm thetime;
    
    time_t clock = time(NULL);
    localtime_r(&clock, &thetime);
    strftime(buf, 1024, "params_%Y%j%H%M%S.txt", &thetime);

    output_file = buf;
    std::cout << "output_file = " << output_file << "\n";

    num_models = 150;

    action = "greedyfit";
    greedy_num_fits = 100;
    greedy_init_iter = 10;
    greedy_refine_iter = 100;
    greedy_replace_iter = 100000;

    max_size = 0;
    preview_size = 512;

    full_iter = 100;
    full_alpha = 4e-5;

    gui = true;

  }

};

void usage(std::ostream& ostr=std::cerr, int code=1) {

  ostr << 
    "usage: imfit [OPTIONS] IMAGEFILE\n"
    "\n"
    "OPTIONS:\n"
    "\n"
    "  -n, --num-models=N       Number of models to fit\n"
    "  -f, --num-fits=N         Number of random guesses per model\n"
    "  -m, --maxiter=N          Maximum # of iterations per trial fit\n"
    "  -r, --refine=N           Maximum # of iterations for final fit\n"
    "  -s, --max-size=N         Maximum size of image to load\n"
    "  -w, --weights=IMGFILE    Load error weights from image\n"
    "  -p, --preview-size=N     Set size of preview\n"
    "  -g, --no-gui             Suppress graphical output\n"
    "  -i, --input=FILE         Read input params from FILE\n"
    "  -o, --output=FILE        Write output params to FILE\n"
    "  -F, --full-iter=NUM      Maximum # of iterations for full refinement\n"
    "  -A, --full-alpha=NUM     Step size for full refinement\n"
    "  -R, --replace-iter=NUM   Maximum # of iterations for replacement\n"
    "  -a, --action=STR         Action to perform (greedyfit,replace)\n"        
    "  -h, --help               See this message\n";

  exit(code);

}

double getdouble(const char* str) {
  char* endptr;
  // strtod: if endptr is not null pointer, sets value to first char after number
  double d = strtod(str, &endptr);
  // so if extra terms, assigned to &endptr 
  if (!endptr || *endptr) {
    std::cerr << "Error parsing number on command line!\n\n";
    // call usage function - give suggestions
    usage();
  }
  return d;
}

/* Converts string to long - also not used?
 * see getdouble for logic
 */

long getlong(const char* str) {
  char* endptr;
  long d = strtol(str, &endptr, 10);
  if (!endptr || *endptr) {
    std::cerr << "Error parsing number on command line!\n\n";
    usage();
  }
  return d;
}

std::string getinputfile(const char* str) {
  std::ifstream istr(str);
  if (!istr.is_open()) {
    std::cerr << "File not found: " << str << "\n\n";
    usage();
  }
  return str;
}

void parse_cmdline(int argc, char** argv, ImFitOptions& opts) {

  const struct option long_options[] = {
    { "num-models",          required_argument, 0, 'n' },
    { "num-fits",            required_argument, 0, 'f' },
    { "maxiter",             required_argument, 0, 'm' },
    { "refine",              required_argument, 0, 'r' },
    { "max-size",            required_argument, 0, 's' },
    { "weights",             required_argument, 0, 'w' },
    { "preview-size",        required_argument, 0, 'p' },
    { "input",               required_argument, 0, 'i' },
    { "output",              required_argument, 0, 'o' },
    { "full-iter",           required_argument, 0, 'f' },
    { "replace-iter",        required_argument, 0, 'R' },
    { "full-alpha",          required_argument, 0, 'A' },
    { "action",              required_argument, 0, 'a' },
    { "no-gui",              no_argument,       0, 'g' },
    { "help",                no_argument,       0, 'h' },
    { 0,                     0,                 0,  0  },
  };

  const char* short_options = "n:f:m:r:s:w:p:i:o:a:F:A:R:gh";

  int opt, option_index;

  while ( (opt = getopt_long(argc, argv, short_options, 
                             long_options, &option_index)) != -1 ) {

    switch (opt) {
    case 'n': opts.num_models = getlong(optarg); break;
    case 'f': opts.greedy_num_fits = getlong(optarg); break;
    case 'm': opts.greedy_init_iter = getlong(optarg); break;
    case 'r': opts.greedy_refine_iter = getlong(optarg); break;
    case 's': opts.max_size = getlong(optarg); break;
    case 'w': opts.weight_file = getinputfile(optarg); break;
    case 'p': opts.preview_size = getlong(optarg); break;
    case 'i': opts.input_file = optarg; break;
    case 'o': opts.output_file = optarg; break;
    case 'a': opts.action = optarg; break;
    case 'F': opts.full_iter = getlong(optarg); break;
    case 'A': opts.full_alpha = getdouble(optarg); break;
    case 'R': opts.greedy_replace_iter = getlong(optarg); break;
    case 'g': opts.gui = false; break;
    case 'h': usage(std::cout, 0); break;
    default: usage(); break;
    }

  }

  if (optind != argc-1) { 
    usage();
  } else {
    opts.image_file = getinputfile(argv[optind]);
  }

}

class FitData {
public:

  double cost;
  std::vector<DMat> outputs;
  std::vector<DMat> params;

  FitData(): cost(DBL_MAX) {}

  FitData(const FitData& f) {
    cost = f.cost;
    for (size_t i=0; i<f.outputs.size(); ++i) {
      outputs.push_back(f.outputs[i].clone());
      params.push_back(f.params[i].clone());
    }
  }
    
  
};

class Fitter {
public:

  ImFitOptions opts;

  cv::Size size;
  cv::Size big_size;

  size_t   n;
  size_t   big_n;

  DMat     target;
  DMat     wmat;

  const double *W;

  DMat     xc, yc, big_xc, big_yc;
  double   px_size;

  Fitter() {}

  void init() {

    //////////////////////////////////////////////////
    // Load images

    UMat gray = cv::imread(opts.image_file, CV_LOAD_IMAGE_GRAYSCALE);

    UMat weights;
    if (!opts.weight_file.empty()) {
      weights = cv::imread(opts.weight_file, CV_LOAD_IMAGE_GRAYSCALE);
      if (weights.size() != gray.size()) {
        std::cerr << "weight file must have same size as image file!\n";
        exit(0);
      }
    }

    //////////////////////////////////////////////////
    // Downsample if necessary

    size = gray.size();

    if (scale_size(size, opts.max_size, SCALE_REDUCE_ONLY, size)) {
      UMat tmp;
      cv::resize(gray, tmp, size, 0, 0, cv::INTER_AREA);
      tmp.copyTo(gray);
      if (!weights.empty()) {
        cv::resize(weights, tmp, size, 0, 0, cv::INTER_AREA);
        tmp.copyTo(weights);
      }
    }
  
    big_size = size;
    scale_size(size, opts.preview_size, SCALE_ALWAYS, big_size);

    //////////////////////////////////////////////////
    // Convert to float
  
    n = size.width * size.height;
    big_n = big_size.width * big_size.height;

    gray.convertTo(target, CV_64F);
    target -= cv::mean(target)[0];
    target /= cv::norm(target.reshape(0, n), cv::NORM_INF);

    W = NULL;

    if (!weights.empty()) {
      weights.convertTo(wmat, CV_64F);
      wmat /= max_element(wmat);
      cv::pow(wmat, 0.5, wmat);
      W = wmat[0];
    }

    //////////////////////////////////////////////////
    // Generate coords, output, problem setup

    gencoords(size, xc, yc);

    px_size = xc(0, 1) - xc(0, 0);

    gencoords(big_size, big_xc, big_yc);

  }

  void load_params(FitData& f) const {

    std::vector<DMat> ptmp;

    std::cout << "at " << __FILE__ << ":" << __LINE__ << "\n";

    std::ifstream istr(opts.input_file.c_str());
    if (!istr.is_open()) {
      std::cerr << "can't read " << opts.input_file << "!\n";
      exit(1);
    }

    std::cout << "at " << __FILE__ << ":" << __LINE__ << "\n";

    size_t psz;
    if (!(istr >> psz)) { 
      std::cerr << "error getting # params\n";
      exit(1);
    }

    std::cout << "at " << __FILE__ << ":" << __LINE__ << "\n";

    for (size_t i=0; i<psz; ++i) {
      DMat pi(GABOR_NUM_PARAMS, 1);
      for (int j=0; j<GABOR_NUM_PARAMS; ++j) {
        if (!(istr >> pi(j))) {
          std::cerr << "error reading!\n";
          exit(1);
        }
      }
      ptmp.push_back(pi);
    }

    std::cout << "at " << __FILE__ << ":" << __LINE__ << "\n";

    f.cost = DBL_MAX;
    f.params.swap(ptmp);
    f.outputs.resize(f.params.size());

    std::cout << "at " << __FILE__ << ":" << __LINE__ << "\n";
    
    DMat output(size, 0.0);

    for (size_t i=0; i<f.params.size(); ++i) {

      f.outputs[i] = DMat(size);
      
      gabor(f.params[i][0], n,
            xc[0], yc[0], (const double*)NULL, (const double*)NULL,
            f.outputs[i][0], (double*)NULL);

      //std::cout << "f.params[" << i << "] = " << f.params[i].t() << "\n";
      //std::cout << "f.outputs[" << i << "] = " << f.outputs[i](cv::Rect(0,0,4,4)) << "\n";
      
      output += f.outputs[i];

    }

    DMat error;

    f.cost = weighted_error(target, output, wmat, error);
    
    std::cout << "init cost is " << f.cost << "\n";

  }

  void save_params(const FitData& f) const {

    std::ofstream ostr(opts.output_file.c_str());
    if (!ostr.is_open()) {
      std::cerr << "can't write " << opts.output_file << "!\n";
      exit(1);
    }
    ostr << f.params.size() << "\n";
    for (size_t i=0; i<f.params.size(); ++i) {
      assert(f.params[i].size() == cv::Size(1, GABOR_NUM_PARAMS));
      for (int j=0; j<GABOR_NUM_PARAMS; ++j) {
        ostr << f.params[i](j) << " ";
      }
      ostr << "\n";
    }
    
    struct stat sb;

    int result = lstat("params_latest.txt", &sb);
    bool do_symlink = false;

    if (result == 0 && S_ISLNK(sb.st_mode)) {
      do_symlink = (unlink("params_latest.txt") == 0);      
    } else if (result < 0 && errno == ENOENT) {
      do_symlink = true;
    }

    if (do_symlink) {
      symlink(opts.output_file.c_str(), "params_latest.txt");
    }

  }

  void full_refine(FitData& f, DMat output, DMat& preview) const {

    const double stepsize = opts.full_alpha / f.params.size();


    for (int iter=0; iter<=opts.full_iter; ++iter) {

      output = DMat(size, 0.0);
      preview = DMat(big_size, 0.0);

      // get outputs
      for (size_t i=0; i<f.outputs.size(); ++i) {

        DMat cur_output(size);
        DMat ptmp(big_size);
        
        gabor(f.params[i][0], n,
              xc[0], yc[0], (const double*)NULL, (const double*)NULL,
              f.outputs[i][0], (double*)NULL);

        output += f.outputs[i];

        if (opts.gui) {

          DMat ptmp(big_size);
          
          gabor(f.params[i][0], big_n, big_xc[0], big_yc[0],
                (const double*) NULL, (const double*)NULL, 
                ptmp[0], (double*)NULL);

          preview += ptmp;

        }

      }

      DMat error;

      // get cost/error vector
      f.cost = weighted_error(target, output, wmat, error);

      std::cout << "cost at full iter " << (iter+1) << " is " << f.cost << "\n";


      if (opts.gui) {

        display(output, preview);

      }

      if (iter == opts.full_iter) {
        break;
      }

      DMat jac(n, GABOR_NUM_PARAMS);

      // take gradient step
      for (size_t i=0; i<f.outputs.size(); ++i) {

        gabor(f.params[i][0], n,
              xc[0], yc[0], W, (const double*)NULL,
              (double*)NULL, jac[0]);

        f.params[i] -= stepsize * (jac.t() * error.reshape(0, n));

      }

    }

  }
  
  double greedy_single(const DMat& rel_target, DMat& best_params) const {

    GaborData gdata(n, xc[0], yc[0], W, rel_target[0]);
    
    double best_cost = -1;
    double info[LM_INFO_SZ];
    
    DMat work(LM_BLEIC_DER_WORKSZ(GABOR_NUM_PARAMS, n, 0, GABOR_NUM_INEQ), 1);

    for (int fit=0; fit<opts.greedy_num_fits; ++fit) {
      
      DMat fit_params(GABOR_NUM_PARAMS, 1);

      if (fit == 0 && !best_params.empty()) {
        best_params.copyTo(fit_params);
      } else {
        gabor_random_params(fit_params[0], px_size, cv::theRNG());
      }

      double final_cost;

      if (opts.greedy_init_iter > 0) {
        final_cost = gdata.fit(fit_params[0], opts.greedy_init_iter, 
                               info, work[0], px_size);
      } else {
        final_cost = gdata(fit_params[0]);
      }

      if (best_cost < 0 || final_cost < best_cost) {
        fit_params.copyTo(best_params);
        best_cost = final_cost;
      }

      std::cout << '.';
      std::cout.flush();

    }

    std::cout << "\n";

    if (opts.greedy_refine_iter > 0) {
      best_cost = gdata.fit(best_params[0], opts.greedy_refine_iter, 
                            info, work[0], px_size);
    }

    return best_cost;

  }

  void greedy_replace(FitData& f, DMat& output, DMat& preview) const {


    for (int iter=0; iter<opts.greedy_replace_iter; ++iter) {

      DMat error;

      size_t replace = cv::theRNG().uniform(0, f.outputs.size());

      output = DMat(size, 0.0);
      preview = DMat(big_size, 0.0);

      for (size_t i=0; i<f.outputs.size(); ++i) {

        if (i != replace) {

          output += f.outputs[i];

          if (opts.gui) {

            DMat ptmp(big_size);
        
            gabor(f.params[i][0], big_n, big_xc[0], big_yc[0], 
                  (const double*)NULL, (const double*)NULL, 
                  ptmp[0], (double*)NULL);
        
            preview += ptmp;

          }

        }

      }

      DMat best_params = f.params[replace].clone();
      DMat rel_target = target - output;

      double best_cost = greedy_single(rel_target, best_params);

      std::cout << "best cost replacing " << replace << " at iter " << (iter+1) << " is " << best_cost << "\n";

      if (best_cost < f.cost) {

        std::cout << "better than previous best cost of " << f.cost << "\n";


        f.cost = best_cost;

        DMat cur_output(size);
        gabor(best_params[0], n, xc[0], yc[0], 
              (double*)NULL, (double*)NULL, cur_output[0], (double*)NULL);
        
        f.outputs[replace] = cur_output;
        f.params[replace] = best_params;

        output += cur_output;

        save_params(f);

        if (opts.gui) {

          DMat pout(big_size);
          gabor(best_params[0], big_n, big_xc[0], big_yc[0],
                (double*)NULL, (double*)NULL, 
                pout[0], (double*)NULL);
        
          preview += pout;

          display(output, preview, cur_output);

        }

      } // if cost improved

    } // for each iter

  } // greedy_replace
  
  void greedy_fit(FitData& f, DMat& output, DMat& preview) const {

    DMat error;

    output = DMat(size, 0.0);
    preview = DMat(big_size, 0.0);

    for (size_t i=0; i<f.outputs.size(); ++i) {

      output += f.outputs[i];

      if (opts.gui) {

        DMat ptmp(big_size);
        
        gabor(f.params[i][0], big_n, big_xc[0], big_yc[0], 
              (const double*)NULL, (const double*)NULL, 
              ptmp[0], (double*)NULL);
        
        preview += ptmp;

      }

    }

    f.cost = weighted_error(target, output, wmat, error, ERR_DISPLAY_POW);

    std::cout << "initial error: " << f.cost << "\n";

    size_t init_model = f.params.size();

    for (size_t model=init_model; model<opts.num_models; ++model) {

      DMat rel_target = target - output;

      DMat best_params;

      double best_cost = greedy_single(rel_target, best_params);

      std::cout << "cost after model " << (model+1) << " is " << best_cost << "\n";

      if (best_cost < f.cost) {

        f.cost = best_cost;

        DMat cur_output(size);
        gabor(best_params[0], n, xc[0], yc[0], 
              (double*)NULL, (double*)NULL, cur_output[0], (double*)NULL);

        f.outputs.push_back(cur_output);
        f.params.push_back(best_params);

        output += cur_output;

        save_params(f);

        if (opts.gui) {

          DMat pout(big_size);
          gabor(best_params[0], big_n, big_xc[0], big_yc[0],
                (double*)NULL, (double*)NULL, 
                pout[0], (double*)NULL);
        
          preview += pout;

          display(output, preview, cur_output);

        }

      }

    }

  }

  void display(const DMat& output,
               const DMat& preview,
               const DMat& cur_output=DMat()) const {

    DMat error;

    weighted_error(target, output, wmat, error, ERR_DISPLAY_POW);

    std::vector<cv::Mat> imgs;
    imgs.push_back(target*0.5 + 0.5);
    imgs.push_back(output*0.5 + 0.5);
    imgs.push_back(error);

    if (!cur_output.empty()) {
      imgs.push_back(cur_output*0.5 + 0.5);
    }

    DMat top = hstack(imgs);

    DMat display(top.rows + preview.rows, 
                 std::max(top.cols, preview.cols), 0.0);

    top.copyTo(display(cv::Rect(0, 0, top.cols, top.rows)));
    display(cv::Rect(0, top.rows, preview.cols, preview.rows)) = 0.5*preview + 0.5;

    cv::imshow("ImFit", display);
    cv::waitKey(5);

  }
               
  

};

/* TODO: make constraints 
     - bounded wavelength (no smaller than X)
     - bounded scale [no smaller than 0.25 wavelength]
*/

int main(int argc, char** argv) {


  Fitter fitter;

#ifdef __APPLE__
  cv::theRNG() = cv::RNG(mach_absolute_time());
#else
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  cv::theRNG() = cv::RNG(ts.tv_nsec);
#endif


  parse_cmdline(argc, argv, fitter.opts);

  fitter.init();


  FitData fdata;

  if (!fitter.opts.input_file.empty()) {
    fitter.load_params(fdata);
  }

  DMat output, preview;

  if (fitter.opts.action == "greedyfit") {
    fitter.greedy_fit(fdata, output, preview);
  } else if (fitter.opts.action == "replace") {
    fitter.greedy_replace(fdata, output, preview);
  }

  if (fitter.opts.gui) {
    fitter.display(output, preview);
    cv::waitKey(0);
  }

  return 0;

}
