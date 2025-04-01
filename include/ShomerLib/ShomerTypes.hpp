#ifndef SHOMER_TYPES_HPP
#define SHOMER_TYPES_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>


/**
 * @namespace shomer_types
 * @brief Contains types and structures used in the Shomer library for image processing and pose estimation.
 */

/**
 * @enum shomer_types::ImageType
 * @brief Represents the type of image format.
 * @var shomer_types::ImageType::BGR
 *      BGR image format.
 * @var shomer_types::ImageType::YUV
 *      YUV image format.
 * @var shomer_types::ImageType::MONO
 *      Monochrome image format.
 */

/**
 * @struct shomer_types::ShomerImage
 * @brief Represents an image with a timestamp.
 * @var shomer_types::ShomerImage::image
 *      The image data stored as a cv::Mat.
 * @var shomer_types::ShomerImage::timestamp
 *      The timestamp associated with the image.
 */

/**
 * @struct shomer_types::CameraPose
 * @brief Represents the pose of a camera.
 * @var shomer_types::CameraPose::R
 *      The rotation matrix (3x3) of the camera.
 * @var shomer_types::CameraPose::t
 *      The translation vector (3x1) of the camera.
 */

/**
 * @struct shomer_types::ShomerPose
 * @brief Represents a camera pose with a timestamp.
 * @var shomer_types::ShomerPose::pose
 *      The camera pose (rotation and translation).
 * @var shomer_types::ShomerPose::timestamp
 *      The timestamp associated with the pose.
 */

/**
 * @struct shomer_types::Blob
 * @brief Represents a 2D point (blob) in an image.
 * @var shomer_types::Blob::x
 *      The x-coordinate of the blob.
 * @var shomer_types::Blob::y
 *      The y-coordinate of the blob.
 */

/**
 * @struct shomer_types::BlobProperties
 * @brief Represents properties of a blob.
 * @var shomer_types::BlobProperties::perimeter
 *      The perimeter of the blob.
 * @var shomer_types::BlobProperties::m00
 *      The zeroth moment (area) of the blob.
 * @var shomer_types::BlobProperties::circularity
 *      The circularity of the blob.
 * @var shomer_types::BlobProperties::hue
 *      The hue value of the blob.
 * @var shomer_types::BlobProperties::boundingContour
 *      The bounding rectangle of the blob's contour.
 */

/**
 * @struct shomer_types::BlobShomer
 * @brief Represents a blob and its associated properties.
 * @var shomer_types::BlobShomer::blob
 *      The blob's 2D coordinates.
 * @var shomer_types::BlobShomer::properties
 *      The properties of the blob.
 */

/**
 * @struct shomer_types::ShomerBlobConfig
 * @brief Configuration for blob detection and filtering.
 * @var shomer_types::ShomerBlobConfig::min_area
 *      Minimum area of a blob to be considered.
 * @var shomer_types::ShomerBlobConfig::max_area
 *      Maximum area of a blob to be considered.
 * @var shomer_types::ShomerBlobConfig::min_circularity
 *      Minimum circularity of a blob to be considered.
 * @var shomer_types::ShomerBlobConfig::saturation_threshold
 *      Saturation threshold for blob detection.
 * @var shomer_types::ShomerBlobConfig::lb_hue
 *      Lower bound for hue filtering.
 * @var shomer_types::ShomerBlobConfig::ub_hue
 *      Upper bound for hue filtering.
 * @var shomer_types::ShomerBlobConfig::circular_mean_hue
 *      Whether to use circular mean for hue calculation.
 */

/**
 * @struct shomer_types::ShomerP4PInfo
 * @brief Configuration for P4P (Perspective-n-Point) problem.
 * @var shomer_types::ShomerP4PInfo::target_points
 *      3D coordinates of the 4 target points.
 * @var shomer_types::ShomerP4PInfo::distortion_coeffs
 *      Distortion coefficients of the camera.
 * @var shomer_types::ShomerP4PInfo::cam_intrinsics
 *      Camera intrinsic parameters.
 */

/**
 * @struct shomer_types::ShomerConfig
 * @brief Configuration for the Shomer library.
 * @var shomer_types::ShomerConfig::p4p_info
 *      Configuration for P4P problem.
 * @var shomer_types::ShomerConfig::blob_config
 *      Configuration for blob detection.
 * @var shomer_types::ShomerConfig::translation_threshold
 *      Threshold for translation changes.
 * @var shomer_types::ShomerConfig::rotation_threshold
 *      Threshold for rotation changes.
 * @var shomer_types::ShomerConfig::max_time_fifo
 *      Maximum time for FIFO queue.
 * @var shomer_types::ShomerConfig::max_distance_lim
 *      Maximum distance limit for filtering.
 * @var shomer_types::ShomerConfig::reject_limit
 *      Limit for rejecting outliers.
 * @var shomer_types::ShomerConfig::kernel_size_gaussian
 *      Kernel size for Gaussian blur.
 * @var shomer_types::ShomerConfig::kernel_size_morph
 *      Kernel size for morphological operations.
 * @var shomer_types::ShomerConfig::it_morph_close
 *      Number of iterations for morphological closing.
 * @var shomer_types::ShomerConfig::it_morph_dilate
 *      Number of iterations for morphological dilation.
 * @var shomer_types::ShomerConfig::image_threshold
 *      Threshold value for image binarization.
 * @var shomer_types::ShomerConfig::filter_size
 *      Size of the filter for image processing.
 * @var shomer_types::ShomerConfig::num_threads
 *      Number of threads for parallel processing.
 * @var shomer_types::ShomerConfig::fisheye
 *      Whether the camera uses a fisheye lens.
 * @var shomer_types::ShomerConfig::radtan
 *      Whether the camera uses radial-tangential distortion.
 * @var shomer_types::ShomerConfig::mono
 *      Whether the image is monochrome.
 * @var shomer_types::ShomerConfig::fifo
 *      Whether to use FIFO queue.
 * @var shomer_types::ShomerConfig::dilate
 *      Whether to apply dilation during processing.
 * @var shomer_types::ShomerConfig::image_type
 *      The type of image format (BGR, YUV, MONO).
 */
namespace shomer_types {


    enum class ImageType {
        BGR = 1,
        YUV = 2,
        MONO = 3
    };


    struct ShomerImage{
        cv::Mat image;
        int64_t timestamp;
    };

    struct CameraPose {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
    };

    struct ShomerPose{
        CameraPose pose;
        int64_t timestamp;
        
    };


    struct Blob {
        double x, y;
    };

    struct BlobProperties {
        double perimeter;
        double m00;
        double circularity;
        double hue;
        cv::Rect boundingContour;
        
    };

    struct BlobShomer { 
        Blob blob;
        BlobProperties properties;
    };

    struct ShomerBlobConfig{
        double min_area;
        double max_area;
        double min_circularity;
        double saturation_threshold;
        double lb_hue;
        double ub_hue;
        bool circular_mean_hue;

    };

struct ShomerP4PInfo {
    std::vector<std::vector<double>> target_points{4, std::vector<double>(3)}; // 4 Target Points
    std::vector<double> distortion_coeffs{4};
    std::vector<double> cam_intrinsics{4};
};
    struct ShomerConfig {
        ShomerP4PInfo p4p_info;

        ShomerBlobConfig blob_config;

        double translation_threshold;
        double rotation_threshold;
        double max_time_fifo;
        double max_distance_lim;

        size_t reject_limit;
        int kernel_size_gaussian;
        int kernel_size_morph;
        int it_morph_close;
        int it_morph_dilate;
        int image_threshold;
        size_t filter_size;
        size_t num_threads;

        bool fisheye;
        bool radtan;
        bool mono;
        bool fifo;
        bool dilate;

        ImageType image_type; //1 bgr, 2 yuv;


    };

} // namespace shomer_types
#endif // SHOMER_TYPES_HPP
