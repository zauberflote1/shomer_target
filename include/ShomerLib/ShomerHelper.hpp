#ifndef SHOMER_HELPER_HPP
#define SHOMER_HELPER_HPP
#include "ShomerUtils.hpp"
#include "ShomerTypes.hpp"

using namespace shomer_types;

class ShomerCore;

/**
 * @class ShomerHelper
 * @brief A helper class for Shomer-related operations, providing utility functions for image processing and point manipulation.
 * 
 * This class is designed to assist with tasks such as undistorting and sorting points, as well as overlaying blobs on images.
 * It works in conjunction with ShomerCore and relies on camera calibration data.
 */
class ShomerHelper {
public:
    /**
     * @brief Constructs a ShomerHelper object with the given P4P information.
     * @param p4p_info_ The P4P information used for initializing the helper.
     */
    ShomerHelper(ShomerP4PInfo p4p_info_);

    /**
     * @brief Destructor for the ShomerHelper class.
     */
    ~ShomerHelper();

protected:
    /**
     * @brief Computes undistorted and sorted points from the given blobs.
     * @param blobs_uv A vector of Blob objects representing points in the image.
     * @param fisheye A boolean indicating whether the camera uses a fisheye lens.
     * @return A 4x2 Eigen matrix containing the undistorted and sorted points.
     */
    Eigen::Matrix<double, 4, 2> getUndistortedSortedPoints(const std::vector<Blob>& blobs_uv, const bool& fisheye);

    /**
     * @brief Generates an image with blobs overlayed on top of the original image.
     * @param image The input image on which blobs will be overlayed.
     * @param blobs A vector of BlobShomer objects representing the blobs to overlay.
     * @return A cv::Mat object containing the overlayed image.
     */
    cv::Mat getOverlayedImage(const cv::Mat& image, const std::vector<BlobShomer>& blobs);

private:
    /**
     * @brief Stores the P4P information used by the helper.
     */
    ShomerP4PInfo p4p_info;

    /**
     * @brief The camera matrix used for camera calibration.
     */
    cv::Mat cameraMatrix_;

    /**
     * @brief The distortion coefficients used for camera calibration.
     */
    cv::Mat distCoeffs_;
};





#endif // SHOMER_HELPER_HPP