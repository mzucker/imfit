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

//////////////////////////////////////////////////////////////////////
// This is a bit of a gross function because it can not only gabor_evaluate
// a single Gabor function, but it can also optionally compute the
// weighted error (difference from target), as well as the Jacobian with
// respect to each Gabor function parameter.
//
// If the weight argument W is NULL, the weights are assumed to be all
// ones.
//
// If the target argument is NULL, the target is assumed to be all
// zeros.
//
// If the jacobian argument is NULL, the Jacobian is not computed.
//
// The function returns one half the sum of squared error values.

inline double gabor(const double* const params, size_t n,
                    const double* x, const double* y, 
                    const double* W, const double* target,
                    double* result,
                    double* jacobian=0) {

  // Get params
  const double& u = params[GABOR_PARAM_U];
  const double& v = params[GABOR_PARAM_V];
  const double& r = params[GABOR_PARAM_R];
  const double& p = params[GABOR_PARAM_P];
  const double& l = params[GABOR_PARAM_L];
  const double& t = params[GABOR_PARAM_T];
  const double& s = params[GABOR_PARAM_S];
  const double& h = params[GABOR_PARAM_H];

  //////////////////////////////////////////////////
  // Compute constants for gabor_evaluating Gabor function
  
  double cr = cos(r);
  double sr = sin(r);

  double f = double(2.0*M_PI)/l;
  double s2 = s*s;
  double t2 = t*t;

  size_t joffs = 0;

  double rval = 0;

  //////////////////////////////////////////////////
  // For each pixel (x,y) location:

  for (size_t i=0; i<n; ++i) {

    //////////////////////////////////////////////////
    // Gabor_Evaluate Gabor function at (x,y)
    
    double xp = double(x[i])-u;
    double yp = double(y[i])-v;

    double b1 = cr*xp + sr*yp;
    double b2 = -sr*xp + cr*yp;

    double b12 = b1*b1;
    double b22 = b2*b2;

    double w = exp(-b12/(double(2.0)*s2) - b22/(double(2.0)*t2));

    double k = f*b1 + p;
    double ck = cos(k);
    double o = h * ck;

    double Wi = W ? W[i] : 1.0;
    double Ti = target ? target[i] : 0.0;

    // Result for pixel i is equal to weight * (Gabor - target)
    double ri = Wi * (w * o - Ti);

    rval += ri * ri;

    // Place into result if result non-NULL
    if (result) {
      result[i] = ri;
    }

    // Compute Jacobian at this pixel if jacobian non-NULL
    if (jacobian) {

      double dw_db1 = -w * b1 / (s*s);
      double dw_db2 = -w * b2 / (t*t);

      double db1_du = -cr;
      double db1_dv = -sr;
      double db1_dr = b2;

      double db2_du = sr;
      double db2_dv = -cr;
      double db2_dr = -b1;

      double dw_du = dw_db1 * db1_du + dw_db2 * db2_du;
      double dw_dv = dw_db1 * db1_dv + dw_db2 * db2_dv;
      double dw_dr = dw_db1 * db1_dr + dw_db2 * db2_dr;
      double dw_ds = w * b12 / (s2*s);
      double dw_dt = w * b22 / (t2*t);

      double dk_db1 = f;
      double dk_dp = 1;
      double dck = -sin(k);
      
      double do_db1 = h * dck * dk_db1;

      double do_dp = h * dck * dk_dp;
      double do_dh = ck;

      double do_du = do_db1 * db1_du;
      double do_dv = do_db1 * db1_dv;
      double do_dr = do_db1 * db1_dr;
      double do_dl = -do_db1 * b1 /l;

      double dg_du = Wi * (dw_du * o + w * do_du);
      double dg_dv = Wi * (dw_dv * o + w * do_dv);
      double dg_dr = Wi * (dw_dr * o + w * do_dr);
      double dg_dl = Wi * (w * do_dl);
      double dg_ds = Wi * (dw_ds * o);
      double dg_dt = Wi * (dw_dt * o);
      double dg_dp = Wi * (w * do_dp);
      double dg_dh = Wi * (w * do_dh);

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

  // Return value.
  return 0.5 * rval;

}

// Wrapper to gabor_evaluate gabor() for levmar library without Jacobian.
// The adata argument will be the ConstantWrapper struct below.
// Read hx as h(x) where h is our nonlinear function for NLS.
inline void gabor_func(double* p, double* hx, int m, int n, void* adata);

// Wrapper to gabor_evaluate gabor() Jacobian for levmar library.
// The adata argument will be the ConstantWrapper struct below.
inline void gabor_jacf(double* p, double*j, int m, int n, void* adata);

//////////////////////////////////////////////////////////////////////
// The ConstantWrapper struct encapsulates the number of pixels, as
// well as all of the constants (x & y coordinates, weights, target).
 
struct ConstantWrapper {
  
  const size_t n;

  const double* x;
  const double* y;
  const double* W;
  const double* target;

  // Constructor
  ConstantWrapper(size_t nn, 
            const double* xx,
            const double* yy, 
            const double* WW,
            const double* tt): n(nn), x(xx), y(yy), W(WW), target(tt) {}

  // Gabor_Evaluate  function
  double gabor_eval(const double* p, double* hx=NULL, double* j=NULL) const {
    return gabor(p, n, x, y, W, target, hx, j);
  }

  // Fit single  function by invoking levmar library.
  double gabor_fit(double* params, int max_iter,
             double info[LM_INFO_SZ], double* work, double px_size) const {

    // Don't need target for levmar library because baked into this object.
    // Similarly default options are fine.
    // Don't care about covariance matrix.
    double* null_target = NULL;
    double* null_opts = NULL;
    double* null_cov = NULL;

    // Const cast is dirty but acceptable here b/c operator() is const.
    void* adata = const_cast<ConstantWrapper*>(this);

    assert( px_size );

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

    return 0.5*info[1];

  }

};

// gabor_func just calls operator() on ConstantWrapper. 
inline void gabor_func(double* p, double* hx, int m, int n, void* adata) {
  assert(m == GABOR_NUM_PARAMS);
  const ConstantWrapper* gcw = static_cast<const ConstantWrapper*>(adata);
  assert(gcw->n == n);
  gcw->gabor_eval(p, hx, NULL);
}

// gabor_jacf just calls operator() on ConstantWrapper.
inline void gabor_jacf(double* p, double*j, int m, int n, void* adata) {
  assert(m == GABOR_NUM_PARAMS);
  const ConstantWrapper* gcw = static_cast<const ConstantWrapper*>(adata);
  assert(gcw->n == n);
  gcw->gabor_eval(p, NULL, j);
}


//////////////////////////////////////////////////////////////////////
// Struct to hold all command-line options.

struct Options {

  std::string image_file;  // positional

  std::string weight_file; // w
  std::string input_file;  // i 
  std::string output_file; // o

  int num_models;          // n
  int greedy_num_fits;     // f
  int greedy_init_iter;    // m
  int greedy_refine_iter;  // r
  int greedy_replace_iter; // R
  int max_size;            // s
  int preview_size;        // p
  int full_iter;           // F
  double full_alpha;       // A
  bool show_gui;           // g

  Options() {

    char buf[1024];

    struct tm thetime;
    
    time_t clock = time(NULL);
    localtime_r(&clock, &thetime);
    strftime(buf, 1024, "params_%Y%j%H%M%S.txt", &thetime);

    output_file = buf;

    num_models = 64;

    greedy_num_fits = 100;
    greedy_init_iter = 10;
    greedy_refine_iter = 100;
    greedy_replace_iter = 100000; // user probably hits Ctrl+C before this happens

    max_size = 32;
    preview_size = 512;

    full_iter = 100;
    full_alpha = 4e-5;

    show_gui = true;

  }

};

//////////////////////////////////////////////////////////////////////
// Show usage options

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
    "  -h, --help               See this message\n";

  exit(code);

}

//////////////////////////////////////////////////////////////////////
// Get real # from command-line arg str

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

//////////////////////////////////////////////////////////////////////
// Get signed integer from command-line arg str

long getlong(const char* str) {
  char* endptr;
  long d = strtol(str, &endptr, 10);
  if (!endptr || *endptr) {
    std::cerr << "Error parsing number on command line!\n\n";
    usage();
  }
  return d;
}

//////////////////////////////////////////////////////////////////////
// Get name of existing file from command-line arg str

std::string getinputfile(const char* str) {
  std::ifstream istr(str);
  if (!istr.is_open()) {
    std::cerr << "File not found: " << str << "\n\n";
    usage();
  }
  return str;
}

//////////////////////////////////////////////////////////////////////
// Parse command line

void parse_cmdline(int argc, char** argv, Options& opts) {

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
    case 'F': opts.full_iter = getlong(optarg); break;
    case 'A': opts.full_alpha = getdouble(optarg); break;
    case 'R': opts.greedy_replace_iter = getlong(optarg); break;
    case 'g': opts.show_gui = false; break;
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

//////////////////////////////////////////////////////////////////////
// Encapsulate parameters and results of Gabor fit.

struct FitData {

  std::vector<DMat> params;
  std::vector<DMat> outputs;
  double cost;

  FitData(): cost(DBL_MAX) {}

  FitData(const FitData& f) {
    cost = f.cost;
    for (size_t i=0; i<f.outputs.size(); ++i) {
      outputs.push_back(f.outputs[i].clone());
      params.push_back(f.params[i].clone());
    }
  }
    
  
};

//////////////////////////////////////////////////////////////////////
// Class to actually do all of the work.

#define ERR_DISPLAY_POW 0.25

class Fitter {
public:

  // Options
  const Options& opts;

  // Image size and preview/output size
  cv::Size size;
  cv::Size preview_size;

  // Products of size and preview_size
  size_t   n;
  size_t   preview_n;

  // Target and weight matrices.
  DMat     target;
  DMat     wmat;

  // W points to either weight matrix or NULL if no weights loaded.
  const double *W;

  // Matrices for x & y coords for both regular and preview.
  DMat     xc, yc, preview_xc, preview_yc;

  // Pixel size (based upon original image size)
  double   px_size;

  Fitter(const Options& o):
    opts(o)
  {

    //////////////////////////////////////////////////
    // Load images

    UMat gray = cv::imread(opts.image_file, CV_LOAD_IMAGE_GRAYSCALE);
    if (gray.empty()) {
      std::cerr << "error reading image from " << opts.image_file << "!\n";
      exit(1);
    }
    std::cout << "loaded image " << opts.image_file
              << " with size " << gray.size() << "\n";

    UMat weights;
    if (!opts.weight_file.empty()) {
      weights = cv::imread(opts.weight_file, CV_LOAD_IMAGE_GRAYSCALE);
      if (weights.size() != gray.size()) {
        std::cerr << "weight file must have same size as image file!\n";
        exit(1);
      }
      std::cout << "loaded weights from " << opts.weight_file << "\n";
    }

    //////////////////////////////////////////////////
    // Downsample if necessary

    size = gray.size();

    if (scale_size(size, opts.max_size, SCALE_REDUCE_ONLY, size)) {
      std::cout << "resizing image to " << size << "\n";
      UMat tmp;
      cv::resize(gray, tmp, size, 0, 0, cv::INTER_AREA);
      tmp.copyTo(gray);
      if (!weights.empty()) {
        cv::resize(weights, tmp, size, 0, 0, cv::INTER_AREA);
        tmp.copyTo(weights);
      }
    }
  
    preview_size = size;
    scale_size(size, opts.preview_size, SCALE_ALWAYS, preview_size);

    //////////////////////////////////////////////////
    // Convert to float
  
    n = size.width * size.height;
    preview_n = preview_size.width * preview_size.height;

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
    gencoords(preview_size, preview_xc, preview_yc);

    px_size = xc(0, 1) - xc(0, 0);
    
  }

  //////////////////////////////////////////////////////////////////////
  // Load params into fit data and fill in outputs and initial cost.

  void load_params(FitData& f) const {

    std::vector<DMat> ptmp;

    std::ifstream istr(opts.input_file.c_str());
    if (!istr.is_open()) {
      std::cerr << "can't read " << opts.input_file << "!\n";
      exit(1);
    }

    size_t psz;
    if (!(istr >> psz)) { 
      std::cerr << "error getting # params\n";
      exit(1);
    }

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

    f.cost = DBL_MAX;
    f.params.swap(ptmp);

    DMat output(size, 0.0);
    compute_initial_cost(f, output);

  }

  void compute_initial_cost(FitData& f, DMat& output) const {
    
    f.outputs.resize(f.params.size());
    
    ConstantWrapper cw(n, xc[0], yc[0], NULL, NULL);

    for (size_t i=0; i<f.params.size(); ++i) {
      f.outputs[i] = DMat(size);
      cw.gabor_eval(f.params[i][0], f.outputs[i][0], NULL);
      output += f.outputs[i];
    }

    DMat error;

    f.cost = weighted_error(target, output, wmat, error);
    
  }

  //////////////////////////////////////////////////////////////////////
  // Save params from fit data

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

  //////////////////////////////////////////////////////////////////////
  // Fit a single gabor function to eliminate the given residual.
  
  double fit_single(const DMat& residual, DMat& best_params) const {

    ConstantWrapper cw(n, xc[0], yc[0], W, residual[0]);
    
    double best_cost = -1;
    double info[LM_INFO_SZ];
    
    DMat work(LM_BLEIC_DER_WORKSZ(GABOR_NUM_PARAMS, n, 0, GABOR_NUM_INEQ), 1);

    //////////////////////////////////////////////////
    // Single fit step 1: try current params (if they exist), as well
    // as a number of random param vectors, pocket the best result.
    // Do a minor amount of refining work for each try.

    for (int fit=0; fit<opts.greedy_num_fits; ++fit) {
      
      DMat fit_params(GABOR_NUM_PARAMS, 1);

      // Only grab current best if exists and first iteration,
      // otherwise choose random.
      if (fit == 0 && !best_params.empty()) {
        best_params.copyTo(fit_params);
      } else {
        gabor_random_params(fit_params[0], px_size, cv::theRNG());
      }

      double final_cost;

      // Do a minor amount of refinement if allowed, otherwise, just
      // get cost.
      if (opts.greedy_init_iter > 0) {
        final_cost = cw.gabor_fit(fit_params[0], opts.greedy_init_iter, 
                                  info, work[0], px_size);
      } else {
        final_cost = cw.gabor_eval(fit_params[0]);
      }

      // If improved, pocket result.
      if (best_cost < 0 || final_cost < best_cost) {
        fit_params.copyTo(best_params);
        best_cost = final_cost;
      }

      std::cout << '.';
      std::cout.flush();

    }

    std::cout << "\n";

    // Do a significant amount of refinement on the best param vector
    // found.
    if (opts.greedy_refine_iter > 0) {
      best_cost = cw.gabor_fit(best_params[0], opts.greedy_refine_iter, 
                               info, work[0], px_size);
    }

    return best_cost;

  }

  //////////////////////////////////////////////////////////////////////
  // Incrementally add models, one-at-a-time, to eliminate residual.
  
  void add_models(FitData& f, DMat& output, DMat& preview) const {

    // Allocate output
    output = DMat(size, 0.0);

    // See if we need to prune instead
    if (f.params.size() > opts.num_models) {

      std::cout << "warning: pruning down to " << opts.num_models << " models!\n";
      cv::Mat_<size_t> idx(f.params.size(), 1);
      for (size_t i=0; i<f.params.size(); ++i) {
        idx(i) = i;
      }

      cv::randShuffle(idx);

      std::vector<DMat> keep_params;
      for (size_t i=0; i<opts.num_models; ++i) {
        size_t j = idx(i);
        keep_params.push_back( f.params[j] );
      }

      keep_params.swap(f.params);

      compute_initial_cost(f, output);

    }

    // Allocate preview
    preview = DMat(preview_size, 0.0);

    
    ConstantWrapper preview_cw(preview_n, preview_xc[0], preview_yc[0],
                               NULL, NULL);
    

    // Compose all outputs and preview
    for (size_t i=0; i<f.outputs.size(); ++i) {

      output += f.outputs[i];

      if (opts.show_gui) {
        DMat ptmp(preview_size);
        preview_cw.gabor_eval(f.params[i][0], ptmp[0], NULL);
        preview += ptmp;
      }

    }

    // Get initial error
    DMat error;
    f.cost = weighted_error(target, output, wmat, error, ERR_DISPLAY_POW);
    std::cout << "initial error: " << f.cost << "\n";

    ConstantWrapper cw(n, xc[0], yc[0], NULL, NULL);

    // For each model we want to add
    for (size_t model=f.params.size(); model<opts.num_models; ++model) {

      // Compute residual
      DMat residual = target - output;

      // Fit parameters
      DMat new_params;
      f.cost = fit_single(residual, new_params);
      std::cout << "error after adding model " << (model+1) << " is " << f.cost << "\n";

      // Get model output
      DMat new_output(size);
      cw.gabor_eval(new_params[0], new_output[0], NULL);

      // Update FitData
      f.outputs.push_back(new_output);
      f.params.push_back(new_params);

      // Update output
      output += new_output;

      // Save params
      save_params(f);

      // Update GUI
      if (opts.show_gui) {

        DMat pout(preview_size);
        
        preview_cw.gabor_eval(best_params[0], pout[0], NULL);
        preview += pout;
        display(output, preview, new_output);
        
      }

    } // for each model to add

  } // add_models


  //////////////////////////////////////////////////////////////////////
  // For a LARGE number of iterations, choose a random model to
  // replace, and see if you can improve it. This is greedy
  // hill-climbing, one model at a time.

  void replace_models(FitData& f, DMat& output, DMat& preview) const {

    // Large number of iterations

    for (int iter=0; iter<opts.greedy_replace_iter; ++iter) {

      DMat error;

      // Choose which model to replace
      size_t replace = cv::theRNG().uniform(0, f.outputs.size());

      output = DMat(size, 0.0);
      preview = DMat(preview_size, 0.0);

      ConstantWrapper preview_cw(preview_n, preview_xc[0], preview_yc[0],
                                 NULL, NULL);

      // Add up outputs of all models but the one to replace
      for (size_t i=0; i<f.outputs.size(); ++i) {

        if (i != replace) {

          output += f.outputs[i];

          if (opts.show_gui) {
            DMat ptmp(preview_size);
            preview_cw.gabor_eval(f.params[i][0], ptmp[0], NULL);
            preview += ptmp;
          }

        }

      }

      // Try replacing it
      DMat best_params = f.params[replace].clone();
      DMat residual = target - output;

      double best_cost = fit_single(residual, best_params);

      std::cout << "best error replacing " << replace << " "
                << "at iter " << (iter+1) << " is " << best_cost << "\n";

      // If improved, update params & display
      if (best_cost < f.cost) {
        
        std::cout << "better than previous best error of " << f.cost << "\n";

        f.cost = best_cost;

        ConstantWrapper cw(n, xc[0], yc[0], NULL, NULL);
        
        DMat cur_output(size);

        cw.gabor_eval(best_params[0], cur_output[0], NULL);
        
        f.outputs[replace] = cur_output;
        f.params[replace] = best_params;

        output += cur_output;

        save_params(f);

        if (opts.show_gui) {

          DMat pout(preview_size);

          preview_cw.gabor_eval(best_params[0], pout[0], NULL);
        
          preview += pout;

          display(output, preview, cur_output);

        }

      } // if cost improved

    } // for each iter

  } // greedy_replace

  //////////////////////////////////////////////////////////////////////
  // Display graphical output/preview

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

//////////////////////////////////////////////////////////////////////
// Main function just strings together all things above.

int main(int argc, char** argv) {

#ifdef __APPLE__
  cv::theRNG() = cv::RNG(mach_absolute_time());
#else
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  cv::theRNG() = cv::RNG(ts.tv_nsec);
#endif

  Options opts;

  parse_cmdline(argc, argv, opts);

  Fitter fitter(opts);

  FitData fdata;

  if (!opts.input_file.empty()) {
    fitter.load_params(fdata);
  }

  DMat output, preview;

  fitter.add_models(fdata, output, preview);

  fitter.replace_models(fdata, output, preview);

  if (opts.show_gui) {
    fitter.display(output, preview);
    cv::waitKey(0);
  }

  return 0;

}
