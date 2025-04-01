#ifndef SHOMER_CERES_HPP
#define SHOMER_CERES_HPP
#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include "ShomerJacobians.hpp"

/**
 * @brief Computes a 3D rotation matrix from Euler angles (phi, theta, psi).
 *
 * This function calculates the rotation matrix by applying three consecutive
 * rotations about the principal axes:
 * - phi: Rotation about the X-axis.
 * - theta: Rotation about the Y-axis.
 * - psi: Rotation about the Z-axis.
 *
 * The resulting rotation matrix is computed as Rz * Ry * Rx, where:
 * - Rx is the rotation matrix for the X-axis.
 * - Ry is the rotation matrix for the Y-axis.
 * - Rz is the rotation matrix for the Z-axis.
 *
 * @param phi Rotation angle (in radians) about the X-axis.
 * @param theta Rotation angle (in radians) about the Y-axis.
 * @param psi Rotation angle (in radians) about the Z-axis.
 * @return Eigen::Matrix3d The resulting 3x3 rotation matrix.
 */
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


/**
 * @class ShomerCeres
 * @brief A Ceres cost function for optimizing camera parameters based on observed and predicted points.
 *
 * This class implements a cost function for use with the Ceres Solver library. It computes the residuals
 * and optionally the Jacobian matrix for a given set of camera parameters, observed 2D points, and a 3D target point.
 *
 * @details
 * The cost function models the projection of a 3D point onto a 2D image plane using a pinhole camera model.
 * The residuals are the differences between the observed 2D points and the projected 2D points.
 *
 * @tparam ceres::SizedCostFunction<2, 6> The cost function computes 2 residuals and takes 6 parameters
 * (3 for translation and 3 for rotation).
 *
 * @note This class uses Eigen for matrix and vector operations and Ceres for rotation computations.
 *
 * @constructor
 * @param observed_point The observed 2D point in the image plane.
 * @param target The 3D target point in the world coordinate system.
 * @param focalx_length The focal length of the camera in the x direction.
 * @param focaly_length The focal length of the camera in the y direction.
 * @param cx The x-coordinate of the principal point in the image plane.
 * @param cy The y-coordinate of the principal point in the image plane.
 *
 * @method Evaluate
 * @brief Computes the residuals and optionally the Jacobian matrix for the given camera parameters.
 * @param parameters A pointer to the array of camera parameters (translation and rotation).
 * @param residuals A pointer to the array where the computed residuals will be stored.
 * @param jacobians A pointer to the array where the computed Jacobian matrix will be stored (if not null).
 * @return True if the computation is successful, false otherwise.
 *
 * @method Create
 * @brief Factory method to create a new instance of the ShomerCeres cost function.
 * @param observed_point The observed 2D point in the image plane.
 * @param target The 3D target point in the world coordinate system.
 * @param focalx_length The focal length of the camera in the x direction.
 * @param focaly_length The focal length of the camera in the y direction.
 * @param cx The x-coordinate of the principal point in the image plane.
 * @param cy The y-coordinate of the principal point in the image plane.
 * @return A pointer to the newly created ShomerCeres cost function.
 *
 * @private_member observed_point_ The observed 2D point in the image plane.
 * @private_member target_ The 3D target point in the world coordinate system.
 * @private_member focalx_length_ The focal length of the camera in the x direction.
 * @private_member focaly_length_ The focal length of the camera in the y direction.
 * @private_member cx_ The x-coordinate of the principal point in the image plane.
 * @private_member cy_ The y-coordinate of the principal point in the image plane.
 */
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