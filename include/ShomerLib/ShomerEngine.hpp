#ifndef SHOMERENGINE_HPP
#define SHOMERENGINE_HPP
#include "ShomerTypes.hpp"
#include <Eigen/Core>


using namespace shomer_types;

class ShomerEngine
{
  
public:
    ShomerEngine(std::vector<double> cam_intrinsics_, std::vector<std::vector<double>> target_);

    ~ShomerEngine();

    CameraPose ShomerEngineSolver(Eigen::Matrix<double, 4, 2>& sorted_observed_pts_undist);  


private:
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    Eigen::Matrix<double, 4, 3> target; //4-TARGET POINTS

};

#endif // SHOMERENGINE_HPP
