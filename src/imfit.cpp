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

/* TODO: make constraints 
     - bounded wavelength (no smaller than X)
     - bounded scale [no smaller than 0.25 wavelength]
*/

int main(int argc, char** argv) {

  ImFitOptions opts;
  parse_cmdline(argc, argv, opts);

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

  cv::Size img_size = gray.size();

  if (scale_size(img_size, opts.max_size, SCALE_REDUCE_ONLY, img_size)) {
    UMat tmp;
    cv::resize(gray, tmp, img_size, 0, 0, cv::INTER_AREA);
    tmp.copyTo(gray);
    if (!weights.empty()) {
      cv::resize(weights, tmp, img_size, 0, 0, cv::INTER_AREA);
      tmp.copyTo(weights);
    }
  }
  
  cv::Size preview_size = img_size;
  scale_size(img_size, opts.preview_size, SCALE_ALWAYS, preview_size);

  //////////////////////////////////////////////////
  // Convert to float
  
  size_t sz = img_size.width * img_size.height;
  size_t psz = preview_size.width * preview_size.height;

  DMat target;
  gray.convertTo(target, CV_64F);
  target -= cv::mean(target)[0];
  target /= cv::norm(target.reshape(0, sz), cv::NORM_INF);

  DMat wmat;
  const double* W = NULL;

  if (!weights.empty()) {
    weights.convertTo(wmat, CV_64F);
    wmat /= max_element(wmat);
    cv::pow(wmat, 0.5, wmat);
    W = wmat[0];
  }

  //////////////////////////////////////////////////
  // Generate coords, output, problem setup

  DMat xc, yc;
  gencoords(img_size, xc, yc);

  double px_size = xc(0, 1) - xc(0, 0);

  DMat xp, yp;
  gencoords(preview_size, xp, yp);

  DMat output(img_size, 0.0);
  DMat preview(preview_size, 0.0);

  DMat work(LM_DER_WORKSZ(GABOR_NUM_PARAMS, sz), 1);

  double info[LM_INFO_SZ];
  double* null_target = 0;
  double* null_opts = 0;
  double* cov = 0;

  cv::RNG rng(mach_absolute_time());

  double prev_cost = DBL_MAX;

  DMat error;

  std::cout << "initial error: " << weighted_error(target, output, wmat, error) << "\n";

  for (int model=0; model<opts.num_models; ++model) {

    DMat rel_target = target - output;
    GaborData gdata(sz, xc[0], yc[0], W, rel_target[0]);

    DMat best_params;
    double best_cost = -1;

    for (int fit=0; fit<opts.num_fits; ++fit) {
      
      DMat params(GABOR_NUM_PARAMS, 1);
      
      gabor_random_params(params[0], rng, px_size);

      double final_cost;

      if (opts.max_iter > 0) {
        final_cost = gdata.fit(params[0], opts.max_iter, info, work[0], px_size);
      } else {
        final_cost = gdata(params[0]);
      }

      if (best_cost < 0 || final_cost < best_cost) {
        params.copyTo(best_params);
        best_cost = final_cost;
      }

      std::cout << '.';
      std::cout.flush();

    }

    std::cout << "\n";

    if (opts.refine_iter > 0) {
      best_cost = gdata.fit(best_params[0], opts.refine_iter, info, work[0], px_size);
    }

    std::cout << "cost after model " << (model+1) << " is " << best_cost << "\n";

    if (best_cost < prev_cost) {

      DMat cur_output(img_size);
      gabor(best_params[0], sz, xc[0], yc[0], 
            (double*)NULL, (double*)NULL, cur_output[0], (double*)NULL);

      output += cur_output;
      prev_cost = best_cost;

      if (opts.gui) {

        DMat werror;
        weighted_error(target, output, wmat, werror);

        for (size_t i=0; i<sz; ++i) { 
          werror[0][i] = sqrt(werror[0][i]); 
        }

        DMat pout(preview_size);
        gabor(best_params[0], psz, xp[0], yp[0],
              (double*)NULL, (double*)NULL, pout[0], (double*)NULL);
        
        preview += pout;

        std::vector<cv::Mat> display;
        display.push_back(target*0.5 + 0.5);
        display.push_back(output*0.5 + 0.5);
        display.push_back(werror);
        display.push_back(cur_output*0.5 + 0.5);

        cv::imshow("foo", hstack(display));
        cv::imshow("preview", 0.5*preview + 0.5);
        cv::waitKey(1);

      }

    }

    

  }

  if (opts.gui) {
    cv::waitKey(0);
  }

  return 0;

}
