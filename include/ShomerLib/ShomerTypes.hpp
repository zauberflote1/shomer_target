#ifndef SHOMER_TYPES_HPP
#define SHOMER_TYPES_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>


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
