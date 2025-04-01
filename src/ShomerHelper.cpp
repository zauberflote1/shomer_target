#include "ShomerHelper.hpp"
//TODO OVERLAY FUNCTION
ShomerHelper::ShomerHelper(ShomerP4PInfo p4p_info_){
    p4p_info = p4p_info_;
    cameraMatrix_ = (cv::Mat_<double>(3, 3) << p4p_info.cam_intrinsics[0], 0, p4p_info.cam_intrinsics[1],
                                                0, p4p_info.cam_intrinsics[2], p4p_info.cam_intrinsics[3],
                                                0, 0, 1);

    distCoeffs_ = (cv::Mat_<double>(1, 4) << p4p_info.distortion_coeffs[0], p4p_info.distortion_coeffs[1], p4p_info.distortion_coeffs[2], p4p_info.distortion_coeffs[3]);
}

ShomerHelper::~ShomerHelper(){
}

Eigen::Matrix<double, 4, 2> ShomerHelper::getUndistortedSortedPoints(const std::vector<Blob>& blobs_uv, const bool& fisheye){
    std::vector<cv::Point2f> distortedPoints;
    distortedPoints.reserve(blobs_uv.size());
    for (const auto& blob : blobs_uv) {
        distortedPoints.emplace_back(blob.x, blob.y);
    }

    std::vector<cv::Point2f> undistortedPoints;
    if (!fisheye) {
        cv::undistortPoints(distortedPoints, undistortedPoints, cameraMatrix_, distCoeffs_);
    } else {
        cv::fisheye::undistortPoints(distortedPoints, undistortedPoints, cameraMatrix_, distCoeffs_);
    }

    std::vector<Eigen::Vector2d> imagePoints;
    imagePoints.reserve(undistortedPoints.size());
    for (const auto& point : undistortedPoints) {
        imagePoints.emplace_back(Eigen::Vector2d(point.x, point.y));
    }

    Eigen::Matrix<double, 4, 2> sortedImagePoints;
    bool success = SortTargetsUsingTetrahedronGeometry(imagePoints, p4p_info.cam_intrinsics, sortedImagePoints);
    if (!success) {
        printf("Failed to sort targets using tetrahedron geometry.");
        return Eigen::Matrix<double, 4, 2>::Zero(); // sortedImagePoints;
    }

    return sortedImagePoints;



}