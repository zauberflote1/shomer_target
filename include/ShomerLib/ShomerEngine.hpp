#ifndef SHOMERENGINE_HPP
#define SHOMERENGINE_HPP
#include "ShomerTypes.hpp"
#include <Eigen/Core>


using namespace shomer_types;

/**
 * @class ShomerEngine
 * @brief A class for solving camera pose estimation problems using a set of observed points and target points.
 * 
 * The ShomerEngine class provides functionality to compute the camera pose given intrinsic camera parameters
 * and a set of observed points. It uses a predefined target configuration for the computation.
 */
class ShomerEngine
{
    /**
     * @brief Constructs a ShomerEngine object with given camera intrinsics and target points.
     * 
     * @param cam_intrinsics_ A vector containing the camera intrinsic parameters [fx, fy, cx, cy].
     * @param target_ A 2D vector representing the 3D coordinates of the target points.
     */
    ShomerEngine(std::vector<double> cam_intrinsics_, std::vector<std::vector<double>> target_);

    /**
     * @brief Destructor for the ShomerEngine class.
     */
    ~ShomerEngine();

    /**
     * @brief Solves for the camera pose using the observed points.
     * 
     * @param sorted_observed_pts_undist A 4x2 matrix of undistorted observed points, sorted in a specific order.
     * @return CameraPose The computed camera pose.
     */
    CameraPose ShomerEngineSolver(Eigen::Matrix<double, 4, 2>& sorted_observed_pts_undist);

private:
    double fx_; ///< Focal length of the camera in the x direction.
    double fy_; ///< Focal length of the camera in the y direction.
    double cx_; ///< Principal point x-coordinate of the camera.
    double cy_; ///< Principal point y-coordinate of the camera.
    Eigen::Matrix<double, 4, 3> target; ///< A 4x3 matrix representing the 3D coordinates of the target points.
};

#endif // SHOMERENGINE_HPP
