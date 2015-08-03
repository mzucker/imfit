#ifdef NDEBUG
#undef NDEBUG
#endif

#include <assert.h>
#include "Timer.h"

//#define USE_CERES

#ifdef USE_CERES
#include "gabor_ceres.h"
#else
#include "gabor.h"
#endif

#include <highgui.h>
#include <cv.h>
#include <time.h>
#include <assert.h>
#include <levmar.h>


int main(int argc, char** argv) {


  cv::Size img_size(100, 75);
  size_t sz = img_size.width * img_size.height;

  DMat xc, yc;

  gencoords(img_size, xc, yc, 1.5);
  
  cv::RNG rng(12345682);
  cv::Mat_<double> params(GABOR_NUM_PARAMS, 1);
  cv::Mat_<double> guess(GABOR_NUM_PARAMS, 1);

  enum {
    NUM_COND = 2
  };

  std::vector<double> total_time(NUM_COND, 0.0);
  std::vector<size_t> num_success(NUM_COND, 0);
  std::vector<size_t> total_iterations(NUM_COND, 0);

  cv::Mat_<double> work(LM_DER_WORKSZ(GABOR_NUM_PARAMS, sz), 1);

  bool quiet = false;
  const char* win = "fitting!";

  if (!quiet) { cv::namedWindow(win); }

  cv::Mat_<double> Wmat(img_size);
  for (int row=0; row<img_size.height; ++row) {
    for (int col=0; col<img_size.width; ++col) {
      double y = yc(row,col);
      double x = xc(row,col);
      Wmat(row, col) = exp(-(x*x + y*y)/(2.0));
    }
  }

  const double* W = Wmat[0];

  for (int iter=0; iter<100; ++iter) {

    cv::Mat_<double> target(img_size), orig(img_size), final(img_size), error(img_size);

    gabor_random_params(params[0], 0.05, rng);
    gabor(params[0], sz, xc[0], yc[0], W, NULL, target[0]);

    gabor_random_params(guess[0], 0.05, rng);

    guess = 0.25 * guess + 0.75 * params;

    gabor(guess[0], sz, xc[0], yc[0], W, NULL, orig[0]);

    GaborData gdata(sz, xc[0], yc[0], W, target[0]);

    if (iter == 0) {

      cv::Mat_<double> J(sz, GABOR_NUM_PARAMS);
      cv::Mat_<double> Jn(sz, GABOR_NUM_PARAMS);

      double* pguess = guess[0];
      double* pJ = J[0];

      gabor_jacf(guess[0], J[0], GABOR_NUM_PARAMS, sz, &gdata);
      double h = 1e-5;

      for (int j=0; j<GABOR_NUM_PARAMS; ++j) {
        double gj = guess(j);
        cv::Mat_<double> ep(sz, 1), en(sz, 1);
        guess(j) = gj + h;
        gabor_func(guess[0], ep[0], GABOR_NUM_PARAMS, sz, &gdata);
        guess(j) = gj - h;
        gabor_func(guess[0], en[0], GABOR_NUM_PARAMS, sz, &gdata);
        guess(j) = gj;
        Jn.col(j) = (ep - en)/(2*h);
      }

      cv::Mat_<double> Jerr;
      cv::absdiff(J, Jn, Jerr);

      cv::Mat_<double> Jmax;
      cv::reduce(Jerr, Jmax, 0, CV_REDUCE_MAX);

      std::cout << "Jmax = " << Jmax << "\n";

      std::cout << "J.row(0) = " << J.row(0) << "\n";
      std::cout << "Jn.row(0) = " << Jn.row(0) << "\n";

      double maxerr = cv::norm(Jmax, cv::NORM_INF);

      std::cout << "maxerr = " << maxerr << "\n";
      assert( maxerr < 1e-6 );


    }

    for (int cond=0; cond<NUM_COND; ++cond) {

      cv::Mat_<double> answer(GABOR_NUM_PARAMS, 1);
      guess.copyTo(answer);

      double time = 0;
      double initial_cost = 0;
      double final_cost = 0;
      size_t iterations = 0;

      if (cond == 0) {

        double info[LM_INFO_SZ];

        Timer t;

        double* null_target = NULL;
        double* opts = NULL;
        double* cov = NULL;

        int result = dlevmar_der(gabor_func, gabor_jacf,
                                 answer[0], null_target,
                                 GABOR_NUM_PARAMS, sz,
                                 50, opts, info, 
                                 work[0], cov, &gdata);
                                 
        time = t.elapsed();

        final_cost = 0.5*info[1];
        initial_cost = 0.5*info[0];
        iterations = info[5];

      } else {
        
#ifdef USE_CERES

        ceres::Problem problem;
        ceres::CostFunction* cost_function =
          new GaborAnalyticCostFunction(sz, xc[0], yc[0], W, target[0]);
        
        problem.AddResidualBlock(cost_function, NULL, answer[0]);

        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = !quiet;
      
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        initial_cost = summary.initial_cost;
        final_cost = summary.final_cost;
        time = summary.total_time_in_seconds;
        iterations = summary.iterations.size();

#else
        
        continue;

#endif

      }

      bool success = final_cost < 1e-6;

      total_time[cond] += time;
      total_iterations[cond] += iterations;
      if (success) { ++num_success[cond]; }


      std::cout << iter << "." << cond << ": " 
                << success << " " << time << " " << iterations << " " 
                << initial_cost << " " << final_cost << "\n";

      if (!quiet) {

        gabor(answer[0], sz, xc[0], yc[0], W, NULL, final[0]);
        gabor(answer[0], sz, xc[0], yc[0], W, target[0], error[0]);

        std::vector<cv::Mat> vec;
        vec.push_back(orig*0.5 + 0.5);
        vec.push_back(final*0.5 + 0.5);
        vec.push_back(target*0.5 + 0.5);
        vec.push_back(error);
        if (W) { vec.push_back(Wmat); }

        cv::Mat display = hstack(vec);

        cv::imshow(win, display);
        cv::waitKey();

        return 0;

      }

    }

  }

  for (int cond=0; cond<NUM_COND; ++cond) {
    std::cout << cond << ": " 
              << num_success[cond] << " " 
              << total_time[cond] << " " 
              << total_iterations[cond] << "\n";
  }

  return 0;

}
