#ifndef SHOMER_CERES_HPP
#define SHOMER_CERES_HPP
#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include "ShomerJacobians.hpp"

inline Eigen::Matrix3d computeRotationMatrix(double phi, double theta, double psi) {
    Eigen::Matrix3d Rx, Ry, Rz;

    Rx << 1, 0, 0,
          0, cos(phi), -sin(phi),
          0, sin(phi), cos(phi);

    Ry << cos(theta), 0, sin(theta),
          0, 1, 0,
          -sin(theta), 0, cos(theta);

    Rz << cos(psi), -sin(psi), 0,
          sin(psi), cos(psi), 0,
          0, 0, 1;

    return Rz * Ry * Rx;
}


class ShomerCeres: public ceres::SizedCostFunction<2, 6> {
public:
    ShomerCeres(const Eigen::Vector2d& observed_point, const Eigen::Vector3d& target, 
                      double focalx_length, double focaly_length, 
                      double cx, double cy)
        : observed_point_(observed_point), target_(target), focalx_length_(focalx_length),
          focaly_length_(focaly_length), cx_(cx), cy_(cy) {}

    virtual ~ShomerCeres() override = default;

    bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override {
        const double* camera_params = parameters[0];
        Eigen::Vector3d camera_T(camera_params[0], camera_params[1], camera_params[2]);
        Eigen::Vector3d camera_R(camera_params[3], camera_params[4], camera_params[5]);
        Eigen::Vector3d known_point_G = target_;

        double camera_point[3];
        double target_point[3] = { target_[0], target_[1], target_[2] };
        ceres::AngleAxisRotatePoint(camera_R.data(), target_point, camera_point);

        camera_point[0] += camera_T[0];
        camera_point[1] += camera_T[1];
        camera_point[2] += camera_T[2];

        double xp = camera_point[0] / camera_point[2];
        double yp = camera_point[1] / camera_point[2];

        Eigen::Vector2d predicted_point;
        predicted_point[0] = xp * focalx_length_ + cx_;
        predicted_point[1] = yp * focaly_length_ + cy_;

        residuals[0] = predicted_point[0] - observed_point_[0];
        residuals[1] = predicted_point[1] - observed_point_[1];

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            Eigen::Matrix<double, 2, 6> J = sym::JacobianMatrix<double>(
                camera_T, camera_R, known_point_G, focalx_length_, focaly_length_, cx_, cy_
            );
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 6; ++j) {
                    jacobians[0][i * 6 + j] = J(i, j);
                }
            }
        }
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d& observed_point, const Eigen::Vector3d& target, 
                                       double focalx_length, double focaly_length, 
                                       double cx, double cy) {
        return new ShomerCeres(observed_point, target, focalx_length, focaly_length, cx, cy);
    }

private:
    Eigen::Vector2d observed_point_;
    Eigen::Vector3d target_;
    double focalx_length_;
    double focaly_length_;
    double cx_;
    double cy_;
};

#endif // SHOMER_CERES_H