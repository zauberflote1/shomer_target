#include "ShomerEngine.hpp"
#include "ShomerCeres.hpp"

ShomerEngine::ShomerEngine(std::vector<double> cam_intrinsics_, std::vector<std::vector<double>> target_){
    assert(cam_intrinsics_.size() >= 4 && "Error: cam_intrinsics_ must have at least 4 elements");

    assert(target_.size() == 4 && "Error: target_ must have exactly 4 rows");
    
    fx_ = cam_intrinsics_[0];
    fy_ = cam_intrinsics_[2];
    cx_ = cam_intrinsics_[1];
    cy_ = cam_intrinsics_[3];

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            target(i, j) = target_[i][j]; // Assign target_[row][col] to target(row, col)
        }
    }
}


ShomerEngine::~ShomerEngine(){
}

CameraPose ShomerEngine::ShomerEngineSolver(Eigen::Matrix<double, 4, 2>& sorted_observed_pts_undist){
    // SET UP CERES PROBLEM
    ceres::Problem problem;

    // SET UP SOLVER OPTIONS
    ceres::Solver::Options options;

    // SETUP INITIAL GUESS
    double opt_params[6] = {0.000, 0.000, -0.001, 0.0, 0.0, 0.7}; //XYZRPY

    // TEMP VECTORS
    Eigen::Vector2d observed_point;
    Eigen::Vector3d target_point;

    // ANALYTIC 
    for (size_t i = 0; i < sorted_observed_pts_undist.rows(); ++i) {  
            observed_point = sorted_observed_pts_undist.row(i).transpose();
            target_point = target.row(i).transpose();

            auto* cost_function = new ShomerCeres(observed_point, target_point, fx_, fy_, cx_, cy_);
            problem.AddResidualBlock(cost_function, nullptr, opt_params);
    }
    options.trust_region_strategy_type = ceres::DOGLEG; 
    options.max_num_iterations = 30;
    options.function_tolerance = 1e-5;
    options.gradient_tolerance = 1e-4 * options.function_tolerance;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;

    // SOLVE IT!
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //GET POSE
    CameraPose pose;
    pose.R = computeRotationMatrix(opt_params[3], opt_params[4], opt_params[5]);
    pose.t = Eigen::Vector3d(opt_params[0], opt_params[1], opt_params[2]);
    return pose;
    }
