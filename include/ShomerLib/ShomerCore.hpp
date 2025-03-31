#ifndef SHOMER_CORE_HPP
#define SHOMER_CORE_HPP
#include <chrono>
#include <vector>
#include <optional>
#include <thread>
#include <future>
#include <mutex>
#include <queue>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <condition_variable>
#include <atomic>

#include "ShomerTypes.hpp"
#include "ShomerHelper.hpp"
#include "ShomerEngine.hpp"

using namespace shomer_types;



class ShomerCore
{
public:
    ShomerCore(const ShomerConfig& options_) 
        : options(options_), engine(options_.p4p_info.cam_intrinsics, options_.p4p_info.target_points), helper(options_.p4p_info) {
        } 


    ~ShomerCore();

    void feed_image(ShomerImage image_);

    std::optional<std::vector<BlobShomer>> getBlobs();
    std::optional<CameraPose> getPose();
    std::optional<cv::Mat> getPreprocessedImage();




private:

    ShomerConfig options;

    ShomerEngine engine;
    
    ShomerHelper helper;

    std::optional<std::vector<BlobShomer>> findAndCalcContoursMono(const cv::Mat &image);

    std::optional<std::vector<BlobShomer>> findAndCalcContours(const cv::Mat &image, const cv::Mat &originalImageHSV);

    cv::Mat preprocessImage(const cv::Mat& image);

    std::vector<BlobShomer> selectBlobs(const std::vector<BlobShomer>& blobs, double min_circularity);
    std::vector<BlobShomer> selectBlobsMono(const std::vector<BlobShomer>& blobs, double min_circularity);
    void processBlobs(const std::vector<Blob>& blobs, const int64_t& timestamp);

    std::vector<BlobShomer> resultBlobs;
    CameraPose result;



    cv::Mat preprocessedImage;


    std::thread process_thread_;

};
#endif
