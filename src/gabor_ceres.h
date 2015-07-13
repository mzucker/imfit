#ifndef _GABOR_CERES_H_
#define _GABOR_CERES_H_

#include <ceres/ceres.h>
#include "gabor.h"

struct GaborResidual {

  const size_t n;
  const double* const x;
  const double* const y;
  const double* const W;
  const double* const target;

  GaborResidual(size_t nn, 
                const double* xx, 
                const double* yy,
                const double* WW, 
                const double* tt):
    n(nn), x(xx), y(yy), W(WW), target(tt) {}

  template <class T>
  bool operator()(T const* const params, T* residual, T* jacobian=0)const {
    gabor(params, n, x, y, W, target, residual, jacobian);
    return true;
  }

};

class GaborAnalyticCostFunction: public ceres::CostFunction {
public:

  GaborResidual res;

  GaborAnalyticCostFunction(size_t nn,
                            const double* xx,
                            const double* yy, 
                            const double* WW, 
                            const double* target):
    res(nn, xx, yy, WW, target)

  {

    std::vector<int>* psz = mutable_parameter_block_sizes();
    assert(psz->empty());
    psz->push_back(GABOR_NUM_PARAMS);

    set_num_residuals(nn);

  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {

    assert(parameter_block_sizes().size() == 1 &&
           parameter_block_sizes()[0] == GABOR_NUM_PARAMS);

    assert(num_residuals() == res.n);

    return res(parameters[0], residuals, 
               jacobians ? jacobians[0] : NULL);

  }

};

typedef ceres::AutoDiffCostFunction<GaborResidual, 
                                    ceres::DYNAMIC, 
                                    GABOR_NUM_PARAMS> GaborCostFunction;

#endif
