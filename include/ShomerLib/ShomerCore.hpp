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



/**
 * @class ShomerCore
 * @brief Core class for processing images, detecting blobs, and computing camera pose.
 *
 * The ShomerCore class is responsible for handling the main pipeline of the Shomer system,
 * including image preprocessing, blob detection, and pose estimation.
 */
class ShomerCore
{
public:
    /**
     * @brief Constructor for ShomerCore.
     * @param options_ Configuration options for the Shomer pipeline.
     */
    ShomerCore(const ShomerConfig& options_);

    /**
     * @brief Destructor for ShomerCore.
     */
    ~ShomerCore();

    /**
     * @brief Feeds an image into the processing pipeline.
     * @param image_ The input image to process.
     */
    void feed_image(ShomerImage image_);

    /**
     * @brief Retrieves the detected blobs from the most recent image processing.
     * @return A std::optional containing a vector of BlobShomer objects if blobs were detected,
     *         or std::nullopt if no blobs were found.
     */
    std::optional<std::vector<BlobShomer>> getBlobs();

    /**
     * @brief Retrieves the computed camera pose from the most recent image processing.
     * @return A std::optional containing a CameraPose object if a valid pose was computed,
     *         or std::nullopt if no valid pose was found.
     */
    std::optional<CameraPose> getPose();

    /**
     * @brief Retrieves the preprocessed version of the most recent input image.
     * @return A std::optional containing a cv::Mat object if preprocessing was successful,
     *         or std::nullopt if no preprocessing was performed.
     */
    std::optional<cv::Mat> getPreprocessedImage();

private:
    /**
     * @brief Detects and calculates blob properties for a monochrome image.
     * @param image The input monochrome image.
     * @return A std::optional containing a vector of BlobShomer objects if blobs were detected,
     *         or std::nullopt if no blobs were found.
     */
    std::optional<std::vector<BlobShomer>> findAndCalcContoursMono(const cv::Mat &image);

    /**
     * @brief Detects and calculates blob properties for a color image.
     * @param image The preprocessed monochrome image.
     * @param originalImageHSV The original color image in HSV format.
     * @return A std::optional containing a vector of BlobShomer objects if blobs were detected,
     *         or std::nullopt if no blobs were found.
     */
    std::optional<std::vector<BlobShomer>> findAndCalcContours(const cv::Mat &image, const cv::Mat &originalImageHSV);

    /**
     * @brief Preprocesses the input image by applying Gaussian blur, thresholding, and morphological operations.
     * @param image The input image to preprocess.
     * @return A cv::Mat object containing the preprocessed image.
     */
    cv::Mat preprocessImage(const cv::Mat& image);

    /**
     * @brief Filters and selects blobs based on circularity and variance for color images.
     * @param blobs A vector of detected BlobShomer objects.
     * @param min_circularity The minimum circularity threshold for blob selection.
     * @return A vector of selected BlobShomer objects.
     */
    std::vector<BlobShomer> selectBlobs(const std::vector<BlobShomer>& blobs, double min_circularity);

    /**
     * @brief Filters and selects blobs based on circularity and variance for monochrome images.
     * @param blobs A vector of detected BlobShomer objects.
     * @param min_circularity The minimum circularity threshold for blob selection.
     * @return A vector of selected BlobShomer objects.
     */
    std::vector<BlobShomer> selectBlobsMono(const std::vector<BlobShomer>& blobs, double min_circularity);

    /**
     * @brief Processes blobs and associates them with a timestamp.
     * @param blobs A vector of detected blobs.
     * @param timestamp The timestamp associated with the blobs.
     */
    void processBlobs(const std::vector<Blob>& blobs, const int64_t& timestamp);

    ShomerConfig options; ///< Configuration options for the Shomer pipeline.
    ShomerEngine engine; ///< Engine for pose estimation and blob processing.
    ShomerHelper helper; ///< Helper for auxiliary operations.

    std::vector<BlobShomer> resultBlobs; ///< Detected blobs from the most recent processing.
    CameraPose result; ///< Computed camera pose from the most recent processing.
    cv::Mat preprocessedImage; ///< Preprocessed version of the most recent input image.

    std::thread process_thread_; ///< Thread for asynchronous processing.
};
#endif
