#ifdef NDEBUG
#undef NDEBUG
#endif

#include <assert.h>

#include "gabor.h"
#include <levmar.h>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <highgui.h>
#include <cv.h>
#include "Timer.h"

#define ERR_DISPLAY_POW 0.25

struct ImFitOptions {

  std::string image_file;
  std::string weight_file;

  int num_models;
  int num_fits;
  int max_iter;
  int refine_iter;
  int max_size;
  int preview_size;
  bool gui;

  ImFitOptions() {
    num_models = 150;
    num_fits = 100;
    max_iter = 10;
    refine_iter = 100;
    max_size = 0;
    preview_size = 512;
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
    { "no-gui",              no_argument,       0, 'g' },
    { "help",                no_argument,       0, 'h' },
    { 0,                     0,                 0,  0  },
  };

  const char* short_options = "n:f:m:r:s:w:p:gh";

  int opt, option_index;

  while ( (opt = getopt_long(argc, argv, short_options, 
                             long_options, &option_index)) != -1 ) {

    switch (opt) {
    case 'n': opts.num_models = getlong(optarg); break;
    case 'f': opts.num_fits = getlong(optarg); break;
    case 'm': opts.max_iter = getlong(optarg); break;
    case 'r': opts.refine_iter = getlong(optarg); break;
    case 's': opts.max_size = getlong(optarg); break;
    case 'w': opts.weight_file = getinputfile(optarg); break;
    case 'p': opts.preview_size = getlong(optarg); break;
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

  std::vector<DMat> outputs;
  std::vector<DMat> params;
  
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


  void multifit(FitData& f) const {

    cv::RNG rng(mach_absolute_time());
    DMat work(LM_DER_WORKSZ(GABOR_NUM_PARAMS, n), 1);

    DMat output(size, 0.0);
    DMat preview(big_size, 0.0);
    DMat error;

    for (size_t i=0; i<f.outputs.size(); ++i) {

      output += f.outputs[i];

      DMat ptmp(big_size);

      gabor(f.params[i][0], big_n, big_xc[0], big_yc[0], W, 
            (const double*)NULL, ptmp[0], (double*)NULL);

      preview += ptmp;

    }

    double prev_cost = weighted_error(target, output, wmat, error, ERR_DISPLAY_POW);

    std::cout << "initial error: " << prev_cost << "\n";

    size_t num_models = opts.num_models - f.outputs.size();

    for (size_t model=0; model<num_models; ++model) {

      DMat rel_target = target - output;
      GaborData gdata(n, xc[0], yc[0], W, rel_target[0]);

      DMat best_params;
      double best_cost = -1;
      double info[LM_INFO_SZ];

      for (int fit=0; fit<opts.num_fits; ++fit) {
      
        DMat fit_params(GABOR_NUM_PARAMS, 1);
      
        gabor_random_params(fit_params[0], rng, px_size);

        double final_cost;

        if (opts.max_iter > 0) {
          final_cost = gdata.fit(fit_params[0], opts.max_iter, info, 
                                 work[0], px_size);
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

      if (opts.refine_iter > 0) {
        best_cost = gdata.fit(best_params[0], opts.refine_iter, 
                              info, work[0], px_size);
      }

      std::cout << "cost after model " << (model+1) << " is " << best_cost << "\n";

      if (best_cost < prev_cost) {

        DMat cur_output(size);
        gabor(best_params[0], n, xc[0], yc[0], 
              (double*)NULL, (double*)NULL, cur_output[0], (double*)NULL);

        output += cur_output;
        prev_cost = best_cost;

        if (opts.gui) {

          DMat pout(big_size);
          gabor(best_params[0], big_n, big_xc[0], big_yc[0],
                (double*)NULL, (double*)NULL, pout[0], (double*)NULL);
        
          preview += pout;

          display(cur_output, output, wmat, error, preview);

        }

      }

    }

  }

  void display(const DMat& cur_output,
               const DMat& output,
               const DMat& wmat, 
               DMat& error,
               const DMat& preview) const {

    weighted_error(target, output, wmat, error, ERR_DISPLAY_POW);

    std::vector<cv::Mat> imgs;
    imgs.push_back(target*0.5 + 0.5);
    imgs.push_back(output*0.5 + 0.5);
    imgs.push_back(error);
    imgs.push_back(cur_output*0.5 + 0.5);

    DMat top = hstack(imgs);

    DMat display(top.rows + preview.rows, 
                 std::max(top.cols, preview.cols), 0.0);

    top.copyTo(display(cv::Rect(0, 0, top.cols, top.rows)));
    display(cv::Rect(0, top.rows, preview.cols, preview.rows)) = 0.5*preview + 0.5;

    cv::imshow("foo", display);
    cv::waitKey(1);

  }
               
  

};

/* TODO: make constraints 
     - bounded wavelength (no smaller than X)
     - bounded scale [no smaller than 0.25 wavelength]
*/

int main(int argc, char** argv) {


  Fitter fitter;

  parse_cmdline(argc, argv, fitter.opts);

  fitter.init();

  FitData fdata;

  fitter.multifit(fdata);

  if (fitter.opts.gui) {
    cv::waitKey(0);
  }

  return 0;

}
