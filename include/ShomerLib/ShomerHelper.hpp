#ifndef SHOMER_HELPER_HPP
#define SHOMER_HELPER_HPP
#include "ShomerUtils.hpp"
#include "ShomerTypes.hpp"

using namespace shomer_types;

class ShomerCore;

class ShomerHelper{
public:
    ShomerHelper(ShomerP4PInfo p4p_info_);
    ~ShomerHelper();

protected:
    friend class ShomerCore;
    Eigen::Matrix<double, 4, 2> getUndistortedSortedPoints(const std::vector<Blob>& blobs_uv, const bool& fisheye);
    cv::Mat getOverlayedImage(const cv::Mat& image, const std::vector<BlobShomer>& blobs);

private:
    ShomerP4PInfo p4p_info;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;



};





#endif // SHOMER_HELPER_HPP