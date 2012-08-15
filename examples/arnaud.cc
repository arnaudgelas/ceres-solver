#include <vector>
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

class CurveCostFunction : public CostFunction
{
  public:
    CurveCostFunction(int num_vertices, double target_length)
      : num_vertices_(num_vertices), target_length_(target_length)
      {
      set_num_residuals(1);
      for (int i = 0; i < num_vertices_; ++i)
        {
        mutable_parameter_block_sizes()->push_back(2);
        }
      }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const
      {
      residuals[0] = target_length_;
      for (int i = 0; i < num_vertices_; ++i)
        {
        int prev = ( num_vertices_ + i - 1 ) % num_vertices_;

        double t = 0.;
        double u = 0.;
        for( int dim = 0; dim < 2; dim++ )
          {
          u = parameters[prev][dim] - parameters[i][dim];
          t += u * u;
          }
        residuals[0] -= sqrt( t );
        }

      if( jacobians )
        {
        // Compute the jacobian blocks and residuals
        for( int i = 0; i < num_vertices_; ++i )
          {
          if( jacobians[i] )
            {
            int prev = ( num_vertices_ + i - 1 ) % num_vertices_;
            int next = ( i + 1 ) % num_vertices_;

            double u[2], v[2];
            double normU = 0., normV = 0.;
            for( int dim = 0; dim < 2; dim++ )
              {
              u[dim] = parameters[i][dim] - parameters[ prev ][dim];
              normU += u[dim] * u[dim];

              v[dim] = parameters[ next ][dim] - parameters[i][dim];
              normV += v[dim] * v[dim];
              }
            normU = sqrt( normU );
            normV = sqrt( normV );

            for( int dim = 0; dim < 2; dim++ )
              {
              jacobians[i][dim] = 0.;

              if( normU > std::numeric_limits< double >::min() )
                {
                jacobians[i][dim] -= u[dim] / normU;
                }
              if( normV > std::numeric_limits< double >::min() )
                {
                jacobians[i][dim] += v[dim] / normV;
                }
              }
            }
          }
        }

        return true;
      }

private:
  int     num_vertices_;
  double  target_length_;
};

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  int N = 100;
  std::vector< double* > y(N);

  const double pi = 3.1415926535897932384626433;

  for( int i = 0; i < N; i++ )
    {
    double theta = i * 2. * pi/ static_cast< double >( N );

    y[i] = new double[2];
    y[i][0] = cos( theta );
    y[i][1] = sin( theta );
    }

  Problem problem;
  problem.AddResidualBlock(new CurveCostFunction( N, 10. ), NULL, y );

  // Run the solver!
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.check_gradients = true;
  options.jacobi_scaling = false;

  Solver::Summary summary;

  std::cout << "Initial " << std::endl;
  for( size_t i = 0; i < y.size(); i++ )
    {
    std::cout << i << " ** " << y[i][0] << " " << y[i][1] << std::endl;
    }

  Solve(options, &problem, &summary);

  std::cout << std::endl;
  std::cout << summary.BriefReport() << std::endl;
  std::cout << std::endl;

  std::cout << "Final " << std::endl;
  for( size_t i = 0; i < y.size(); i++ )
    {
    std::cout << i << " ** " << y[i][0] << " " << y[i][1] << std::endl;
    delete[] y[i];
    }

  return 0;
}
